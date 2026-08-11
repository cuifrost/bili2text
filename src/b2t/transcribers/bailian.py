from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from b2t.progress import ProgressReporter
from b2t.transcribers.base import Transcriber


class BailianFileTranscriber(Transcriber):
    """Submit long audio to Alibaba Cloud Bailian's async file transcription API."""

    name = "bailian-filetrans"
    accepts_original_audio = True

    def __init__(
        self,
        *,
        api_key: str = "",
        workspace_id: str = "",
        region: str = "cn-beijing",
        model_name: str = "qwen3-asr-flash-filetrans",
        language: str | None = "zh",
        storage_mode: str = "auto",
        oss_region: str = "cn-beijing",
        oss_bucket: str = "",
        oss_endpoint: str = "",
        oss_prefix: str = "bili2text/audio",
        oss_url_expire_seconds: int = 86400,
        poll_interval_seconds: float = 3.0,
        poll_timeout_seconds: int = 7200,
        cleanup_uploaded_audio: bool = True,
        enable_itn: bool = False,
        enable_words: bool = True,
        request_retry_count: int = 3,
        retry_backoff_seconds: float = 2.0,
        session: Any | None = None,
        oss_client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = _env_first("DASHSCOPE_API_KEY", api_key)
        self.workspace_id = _env_first("DASHSCOPE_WORKSPACE_ID", workspace_id)
        self.region = _env_first("DASHSCOPE_REGION", region, "cn-beijing")
        self.model_name = _env_first("B2T_BAILIAN_MODEL", model_name, "qwen3-asr-flash-filetrans")
        self.language = language
        self.storage_mode = _env_first("B2T_BAILIAN_STORAGE", storage_mode, "auto").lower()
        if self.storage_mode not in {"auto", "temporary", "oss"}:
            raise ValueError("B2T_BAILIAN_STORAGE must be auto, temporary, or oss")
        self.temporary_upload_endpoint = _env_first(
            "DASHSCOPE_UPLOAD_ENDPOINT",
            "https://dashscope.aliyuncs.com/api/v1/uploads",
        )
        self.oss_region = _env_first("OSS_REGION", oss_region, self.region)
        self.oss_bucket = _env_first("OSS_BUCKET", oss_bucket)
        self.oss_endpoint = _env_first("OSS_ENDPOINT", oss_endpoint)
        self.oss_prefix = _env_first("OSS_PREFIX", oss_prefix, "bili2text/audio")
        self.oss_url_expire_seconds = max(300, int(oss_url_expire_seconds))
        self.poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self.poll_timeout_seconds = max(30, int(poll_timeout_seconds))
        self.cleanup_uploaded_audio = cleanup_uploaded_audio
        self.enable_itn = _env_bool("B2T_BAILIAN_ENABLE_ITN", enable_itn)
        self.enable_words = _env_bool("B2T_BAILIAN_ENABLE_WORDS", enable_words)
        self.request_retry_count = max(1, int(request_retry_count))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._session = session
        self._oss_client = oss_client
        self._sleep = sleep_fn

    def transcribe(
        self,
        audio_path: Path,
        *,
        prompt: str | None = None,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        self._validate_configuration()
        session = self._get_session()
        object_key: str | None = None
        storage = "temporary"

        try:
            if progress is not None:
                progress.running(
                    "transcribing",
                    message="uploading_audio",
                    indeterminate=True,
                )
            file_url, object_key, storage = self._upload_audio(audio_path, session)

            if progress is not None:
                progress.running(
                    "transcribing",
                    message="submitting_task",
                    indeterminate=True,
                )
            task_id = self._submit_task(session, file_url, prompt)

            if progress is not None:
                progress.running(
                    "transcribing",
                    message="waiting_remote",
                    indeterminate=True,
                    detail={"task_id": task_id},
                )
            task_output = self._wait_for_task(session, task_id, progress)
            result_url = _find_transcription_url(task_output)
            if not result_url:
                raise RuntimeError("Bailian task completed without a transcription URL")

            result_response = self._request(session, "get", result_url, timeout=120)
            result_data = _response_json(result_response, "Bailian transcription result download failed")
            text, segments, language = _parse_transcription_result(result_data)
            if not text:
                raise RuntimeError("Bailian returned an empty transcript")

            return {
                "text": text,
                "segments": segments,
                "language": language or self.language,
                "model": self.model_name,
                "metadata": {
                    "task_id": task_id,
                    "oss_object_key": object_key,
                    "storage": storage,
                    "provider": self.name,
                },
            }
        finally:
            if object_key and self.cleanup_uploaded_audio:
                self._delete_audio(object_key)

    def _validate_configuration(self) -> None:
        missing: list[str] = []
        if not self.api_key:
            missing.append("DASHSCOPE_API_KEY")
        if not self.workspace_id:
            missing.append("DASHSCOPE_WORKSPACE_ID")
        if self._uses_oss_storage():
            if not self.oss_bucket:
                missing.append("OSS_BUCKET")
            if not os.getenv("OSS_ACCESS_KEY_ID"):
                missing.append("OSS_ACCESS_KEY_ID")
            if not os.getenv("OSS_ACCESS_KEY_SECRET"):
                missing.append("OSS_ACCESS_KEY_SECRET")
        if missing:
            raise RuntimeError(
                "Bailian file transcription is missing configuration: "
                + ", ".join(missing)
                + ". Set these environment variables or run `bili2text bootstrap`."
            )

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Bailian support is not installed. "
                "Run `uv sync --extra bailian` to install it."
            ) from exc
        self._session = requests.Session()
        return self._session

    def _uses_oss_storage(self) -> bool:
        return self.storage_mode == "oss" or (self.storage_mode == "auto" and bool(self.oss_bucket))

    def _upload_audio(self, audio_path: Path, session: Any) -> tuple[str, str | None, str]:
        if self._uses_oss_storage():
            file_url, object_key = self._upload_to_oss(audio_path)
            return file_url, object_key, "oss"
        return self._upload_to_temporary(audio_path, session), None, "bailian-temporary"

    def _upload_to_oss(self, audio_path: Path) -> tuple[str, str]:
        try:
            import alibabacloud_oss_v2 as oss
        except ImportError as exc:
            raise RuntimeError(
                "Bailian support is not installed. "
                "Run `uv sync --extra bailian` to install it."
            ) from exc

        client = self._get_oss_client(oss)
        suffix = audio_path.suffix.lower() or ".wav"
        timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        object_key = f"{self.oss_prefix.rstrip('/')}/{timestamp}/{uuid.uuid4().hex}{suffix}"
        put_request = oss.PutObjectRequest(bucket=self.oss_bucket, key=object_key)
        client.uploader().upload_file(put_request, filepath=str(audio_path))

        get_request = oss.GetObjectRequest(bucket=self.oss_bucket, key=object_key)
        presigned = client.presign(
            get_request,
            expires=timedelta(seconds=self.oss_url_expire_seconds),
        )
        return str(presigned.url), object_key

    def _upload_to_temporary(self, audio_path: Path, session: Any) -> str:
        policy_response = self._request(
            session,
            "get",
            self.temporary_upload_endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            params={"action": "getPolicy", "model": self.model_name},
            timeout=60,
        )
        policy_payload = _response_json(policy_response, "Bailian temporary upload policy request failed")
        policy_data = policy_payload.get("data")
        if not isinstance(policy_data, dict):
            raise RuntimeError("Bailian temporary upload policy response has no data object")

        required_fields = (
            "upload_host",
            "upload_dir",
            "oss_access_key_id",
            "signature",
            "policy",
            "x_oss_object_acl",
            "x_oss_forbid_overwrite",
        )
        missing = [
            field
            for field in required_fields
            if field not in policy_data or policy_data[field] is None or policy_data[field] == ""
        ]
        if missing:
            raise RuntimeError(
                "Bailian temporary upload policy is missing fields: " + ", ".join(missing)
            )

        file_name = audio_path.name
        object_key = f"{str(policy_data['upload_dir']).rstrip('/')}/{file_name}"
        with audio_path.open("rb") as file_handle:
            response = self._request(
                session,
                "post",
                str(policy_data["upload_host"]),
                files={
                    "OSSAccessKeyId": (None, str(policy_data["oss_access_key_id"])),
                    "Signature": (None, str(policy_data["signature"])),
                    "policy": (None, str(policy_data["policy"])),
                    "x-oss-object-acl": (None, str(policy_data["x_oss_object_acl"])),
                    "x-oss-forbid-overwrite": (None, str(policy_data["x_oss_forbid_overwrite"])),
                    "key": (None, object_key),
                    "success_action_status": (None, "200"),
                    "file": (file_name, file_handle),
                },
                timeout=300,
            )
        if response.status_code < 200 or response.status_code >= 300:
            detail = getattr(response, "text", "") or response.status_code
            raise RuntimeError(f"Bailian temporary audio upload failed: {detail}")
        return f"oss://{object_key}"

    def _get_oss_client(self, oss: Any) -> Any:
        if self._oss_client is not None:
            return self._oss_client
        config = oss.config.load_default()
        config.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        config.region = self.oss_region
        if self.oss_endpoint:
            config.endpoint = self.oss_endpoint
        self._oss_client = oss.Client(config)
        return self._oss_client

    def _delete_audio(self, object_key: str) -> None:
        try:
            import alibabacloud_oss_v2 as oss

            client = self._get_oss_client(oss)
            client.delete_object(oss.DeleteObjectRequest(bucket=self.oss_bucket, key=object_key))
        except Exception:
            return

    def _submit_task(self, session: Any, file_url: str, prompt: str | None) -> str:
        parameters: dict[str, Any] = {
            "channel_id": [0],
            "enable_itn": self.enable_itn,
            "enable_words": self.enable_words,
        }
        normalized_language = _normalize_language(self.language)
        if normalized_language:
            parameters["language"] = normalized_language
        if prompt:
            parameters["corpus"] = {"text": prompt}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        if file_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"

        response = self._request(
            session,
            "post",
            self._api_url("/services/audio/asr/transcription"),
            headers=headers,
            json={
                "model": self.model_name,
                "input": {"file_url": file_url},
                "parameters": parameters,
            },
            timeout=60,
        )
        data = _response_json(response, "Bailian transcription submission failed")
        task_id = str((data.get("output") or {}).get("task_id") or data.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"Bailian submission returned no task id: {data}")
        return task_id

    def _wait_for_task(
        self,
        session: Any,
        task_id: str,
        progress: ProgressReporter | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.poll_timeout_seconds
        while time.monotonic() < deadline:
            response = self._request(
                session,
                "get",
                self._api_url(f"/tasks/{task_id}"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable",
                },
                timeout=60,
            )
            data = _response_json(response, "Bailian task polling failed")
            output = data.get("output") or {}
            status = str(output.get("task_status") or output.get("status") or data.get("task_status") or "").upper()
            if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                return data
            if status in {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}:
                message = output.get("message") or output.get("code") or data
                raise RuntimeError(f"Bailian task failed: {message}")
            if progress is not None:
                progress.running(
                    "transcribing",
                    message="waiting_remote",
                    indeterminate=True,
                    detail={"task_id": task_id, "status": status or "RUNNING"},
                )
            self._sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Bailian task timed out after {self.poll_timeout_seconds} seconds: {task_id}")

    def _request(self, session: Any, method: str, url: str, **kwargs: Any) -> Any:
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.request_retry_count):
            try:
                response = getattr(session, method)(url, **kwargs)
                if response.status_code not in retryable_statuses or attempt == self.request_retry_count - 1:
                    return response
            except Exception as exc:
                last_error = exc
                if attempt == self.request_retry_count - 1:
                    raise RuntimeError(f"Bailian request failed: {exc}") from exc
            self._sleep(self.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(f"Bailian request failed: {last_error}") from last_error
        raise RuntimeError("Bailian request failed without a response")

    def _api_url(self, suffix: str) -> str:
        host = f"{self.workspace_id}.{self.region}.maas.aliyuncs.com"
        return f"https://{host}/api/v1{suffix}"


def _response_json(response: Any, message: str) -> dict[str, Any]:
    if response.status_code < 200 or response.status_code >= 300:
        detail = getattr(response, "text", "") or response.status_code
        raise RuntimeError(f"{message}: {detail}")
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"{message}: invalid JSON response") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{message}: response is not an object")
    if data.get("code") and data.get("code") not in {"200", "OK"}:
        raise RuntimeError(f"{message}: {data.get('code')} {data.get('message') or data.get('msg') or ''}".strip())
    return data


def _find_transcription_url(data: dict[str, Any]) -> str | None:
    output = data.get("output") or {}
    result = output.get("result") or {}
    candidates = [
        result.get("transcription_url"),
        output.get("transcription_url"),
        data.get("transcription_url"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)

    results = output.get("results") or data.get("results") or []
    if isinstance(results, dict):
        results = [results]
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("transcription_url"):
                return str(item["transcription_url"])
    return None


def _parse_transcription_result(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    transcripts = data.get("transcripts") or data.get("results") or []
    if isinstance(transcripts, dict):
        transcripts = [transcripts]

    text_parts: list[str] = []
    segments: list[dict[str, Any]] = []
    language: str | None = None
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            continue
        language = language or transcript.get("language")
        sentences = transcript.get("sentences") or transcript.get("segments") or []
        if isinstance(sentences, list) and sentences:
            for sentence in sentences:
                if not isinstance(sentence, dict):
                    continue
                sentence_text = str(sentence.get("text") or "").strip()
                if sentence_text:
                    text_parts.append(sentence_text)
                start_value = sentence.get("begin_time")
                if start_value is None:
                    start_value = sentence.get("start")
                end_value = sentence.get("end_time")
                if end_value is None:
                    end_value = sentence.get("end")
                segments.append(
                    {
                        "start": _milliseconds_to_seconds(start_value),
                        "end": _milliseconds_to_seconds(end_value),
                        "text": sentence_text,
                    }
                )
        else:
            transcript_text = str(transcript.get("text") or "").strip()
            if transcript_text:
                text_parts.append(transcript_text)

    text = "\n".join(text_parts).strip()
    if not text:
        text = str(data.get("text") or "").strip()
    return text, segments, language


def _milliseconds_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric / 1000 if numeric > 100 else numeric


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower().replace("_", "-")
    if normalized.startswith("zh"):
        return "zh"
    return normalized.split("-", 1)[0]


def _env_first(name: str, value: str, default: str = "") -> str:
    environment_value = os.getenv(name, "").strip()
    return environment_value or value.strip() or default


def _env_bool(name: str, value: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return value
    return raw.strip().lower() in {"1", "true", "yes", "on"}
