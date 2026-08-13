"""ComfyUI HTTP endpoint for ten-image ZIP imports."""

import asyncio
import pathlib
import tempfile
import zipfile

import folder_paths

from .batch_import import MAX_ARCHIVE_BYTES, extract_image_zip


def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    @PromptServer.instance.routes.post("/wan-loop/batch/import-zip")
    async def import_zip(request):
        temporary_name = None
        try:
            reader = await request.multipart()
            field = await reader.next()
            if field is None or field.name != "archive":
                raise ValueError("multipart field 'archive' is required")
            suffix = pathlib.Path(field.filename or "batch.zip").suffix.lower()
            if suffix != ".zip":
                raise ValueError("only .zip archives are supported")

            size = 0
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
                temporary_name = temporary.name
                while True:
                    chunk = await field.read_chunk(size=1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("ZIP is larger than the 300 MiB import limit")
                    temporary.write(chunk)

            result = await asyncio.to_thread(
                extract_image_zip,
                temporary_name,
                folder_paths.get_input_directory(),
            )
            return web.json_response(result)
        except (ValueError, OSError, zipfile.BadZipFile, UnicodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        finally:
            if temporary_name:
                pathlib.Path(temporary_name).unlink(missing_ok=True)


# Register once when ComfyUI imports this custom-node package.
register_routes()
