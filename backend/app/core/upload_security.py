"""上传文件的内容级安全校验。"""

from __future__ import annotations

import struct
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


class UploadSecurityError(ValueError):
    """上传内容不可信或超出解析安全边界。"""


class MalwareScannerUnavailable(RuntimeError):
    """启用恶意文件扫描后，扫描器不可用。"""


def validate_uploaded_file(path: Path, extension: str, settings) -> None:
    """校验扩展名与魔数、容器膨胀、PDF页数/加密和图片像素。"""
    header = path.read_bytes()[:32]
    ext = extension.lower()
    if ext == ".pdf":
        if not header.startswith(b"%PDF-"):
            raise UploadSecurityError("PDF 文件签名不匹配")
        _validate_pdf(path, settings.max_pdf_pages)
    elif ext == ".png":
        if not header.startswith(b"\x89PNG\r\n\x1a\n"):
            raise UploadSecurityError("PNG 文件签名不匹配")
        width, height = struct.unpack(">II", header[16:24])
        _validate_pixels(width, height, settings.max_image_pixels)
    elif ext in {".jpg", ".jpeg"}:
        if not header.startswith(b"\xff\xd8\xff"):
            raise UploadSecurityError("JPEG 文件签名不匹配")
        width, height = _jpeg_dimensions(path)
        _validate_pixels(width, height, settings.max_image_pixels)
    elif ext in {".docx", ".xlsx"}:
        if not header.startswith(b"PK\x03\x04"):
            raise UploadSecurityError("Office 文件签名不匹配")
        _validate_office_archive(path, ext, settings)
    else:
        raise UploadSecurityError("不支持的文件类型")

    if settings.clamav_scan_enabled:
        _scan_with_clamav(path, settings.clamav_command, settings.clamav_timeout)


def _validate_pdf(path: Path, max_pages: int) -> None:
    import fitz

    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise UploadSecurityError("不支持加密或需要密码的 PDF")
            if document.page_count <= 0:
                raise UploadSecurityError("PDF 不包含可解析页面")
            if document.page_count > max_pages:
                raise UploadSecurityError(f"PDF 页数超过限制 {max_pages}")
    except UploadSecurityError:
        raise
    except Exception as exc:
        raise UploadSecurityError("PDF 结构损坏或无法解析") from exc


def _validate_office_archive(path: Path, extension: str, settings) -> None:
    required = "word/document.xml" if extension == ".docx" else "xl/workbook.xml"
    max_uncompressed = settings.max_archive_uncompressed_mb * 1024 * 1024
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > settings.max_archive_entries:
                raise UploadSecurityError("Office 文件包含过多归档条目")
            names = {info.filename for info in infos}
            if required not in names:
                raise UploadSecurityError("Office 文件结构与扩展名不匹配")
            total_uncompressed = 0
            for info in infos:
                normalized = PurePosixPath(info.filename.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise UploadSecurityError("Office 文件包含不安全路径")
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed:
                    raise UploadSecurityError("Office 文件解压后超过大小限制")
                compressed = max(info.compress_size, 1)
                if info.file_size / compressed > settings.max_archive_ratio:
                    raise UploadSecurityError("Office 文件压缩比异常")
    except UploadSecurityError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise UploadSecurityError("Office 文件结构损坏") from exc


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise UploadSecurityError("JPEG 文件签名不匹配")
        while True:
            byte = stream.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker or marker in {b"\xd8", b"\xd9"}:
                continue
            length_raw = stream.read(2)
            if len(length_raw) != 2:
                break
            segment_length = int.from_bytes(length_raw, "big")
            if segment_length < 2:
                break
            if marker[0] in sof_markers:
                data = stream.read(5)
                if len(data) != 5:
                    break
                return int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")
            stream.seek(segment_length - 2, 1)
    raise UploadSecurityError("无法读取 JPEG 尺寸")


def _validate_pixels(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise UploadSecurityError("图片尺寸非法")
    if width * height > max_pixels:
        raise UploadSecurityError(f"图片像素超过限制 {max_pixels}")


def _scan_with_clamav(path: Path, command: str, timeout: int) -> None:
    try:
        result = subprocess.run(
            [command, "--no-summary", str(path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MalwareScannerUnavailable("恶意文件扫描服务不可用") from exc
    if result.returncode == 1:
        raise UploadSecurityError("文件未通过恶意内容扫描")
    if result.returncode != 0:
        raise MalwareScannerUnavailable("恶意文件扫描服务不可用")
