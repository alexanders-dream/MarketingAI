import httpx

class UploadPostClient:
    """Client for connecting to Upload-Post universal social media API"""
    
    BASE_URL = "https://upload-post.com/api"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def create_user(self, external_id: str, email: str) -> dict:
        """Create a white-label user or get existing"""
        # Upload-post POST /api/uploadposts/users
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE_URL}/uploadposts/users",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"external_id": external_id, "email": email}
            )
            return resp.json()

    async def get_jwt(self, user_id: str) -> str:
        """Get an authentication JWT for creating the OAuth popup"""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE_URL}/uploadposts/users/generate-jwt",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"user_id": user_id}
            )
            return resp.json().get("token")

    async def upload_photo_by_url(self, payload: dict) -> dict:
        """Upload photo using text or image public URL"""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE_URL}/upload_photos",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            return resp.json()

    async def upload_photo_bytes(self, image_bytes: bytes, platforms: list, caption: str) -> dict:
        """Upload photo bytes from a private media source"""
        async with httpx.AsyncClient() as client:
            files = {'files': ('image.jpg', image_bytes, 'image/jpeg')}
            data = {'platforms': ",".join(platforms), 'caption': caption}
            resp = await client.post(
                f"{self.BASE_URL}/upload_photos",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files
            )
            return resp.json()

    async def get_analytics(self, profile_id: str) -> dict:
        """Get analytics for connected profile"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE_URL}/analytics/{profile_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return resp.json()
