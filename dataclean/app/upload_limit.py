from fastapi import HTTPException
from starlette.responses import JSONResponse


class UploadBodyLimit:
    """Bound multipart parsing too, before UploadFile is fully spooled to disk."""
    def __init__(self, app, max_file_bytes):
        self.app = app
        # Reserve 64 KiB for the multipart boundary, filename and form headers.
        self.limit = max_file_bytes + 64 * 1024

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/jobs":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length and (not length.isdigit() or int(length) > self.limit):
            return await JSONResponse({"detail": "上传请求过大"}, status_code=413)(scope, receive, send)
        size = 0

        async def counted_receive():
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.limit:
                    raise HTTPException(413, "上传请求过大")
            return message

        return await self.app(scope, counted_receive, send)
