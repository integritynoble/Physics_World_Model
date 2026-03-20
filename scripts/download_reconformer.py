"""Download ReconFormer checkpoint from JHU SharePoint OneDrive."""
import requests
import re
import os
import sys
import urllib.parse

URL = "https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/Er37oIyNy3NBrXbeCQBp_fQBAxELR8UDaq6gHd-fjwRrSw"
POST_URL = "https://livejohnshopkins-my.sharepoint.com/personal/pguo4_jh_edu/_layouts/15/guestaccess.aspx?share=Er37oIyNy3NBrXbeCQBp_fQBAxELR8UDaq6gHd-fjwRrSw"
PASSWORD = "pguo4@jhu.edu"
OUT_DIR = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/reference/mri"
BASE = "https://livejohnshopkins-my.sharepoint.com"

os.makedirs(OUT_DIR, exist_ok=True)

session = requests.Session()

# Step 1: GET the password page
print("Step 1: Fetching password page...")
resp = session.get(URL, allow_redirects=True)
html = resp.text

# Step 2: Extract form fields and POST password
fields = {}
for name in ['__VIEWSTATE', '__VIEWSTATEGENERATOR', '__VIEWSTATEENCRYPTED', '__EVENTVALIDATION']:
    m = re.search(r'name="' + name + r'"[^>]*value="([^"]*)"', html)
    fields[name] = m.group(1) if m else ""

print("Step 2: Submitting password...")
data = {
    **fields,
    "txtPassword": PASSWORD,
    "btnSubmitPassword": "Verify",
    "__EVENTTARGET": "btnSubmitPassword",
    "__EVENTARGUMENT": "",
}

resp2 = session.post(POST_URL, data=data, allow_redirects=True)
if "Enter Password" in resp2.text:
    print("  ERROR: Password rejected.")
    sys.exit(1)
print("  Password accepted!")

# Step 3: List files via REST API
folder_path = "/personal/pguo4_jh_edu/Documents/ReconFormer/checkpoints"
list_url = f"{BASE}/personal/pguo4_jh_edu/_api/web/GetFolderByServerRelativeUrl('{urllib.parse.quote(folder_path)}')/Files"

print("\nStep 3: Listing files...")
resp3 = session.get(list_url, headers={"Accept": "application/json;odata=verbose"})
files = resp3.json().get("d", {}).get("results", [])
print(f"  Found {len(files)} files:")
for f in files:
    name = f.get("Name", "?")
    size = int(f.get("Length", 0))
    server_url = f.get("ServerRelativeUrl", "")
    print(f"    {name} ({size / 1024 / 1024:.1f} MB) -> {server_url}")

# Step 4: Download each .pth file
print("\nStep 4: Downloading checkpoint files...")
for f in files:
    name = f.get("Name", "")
    if not name.endswith(".pth"):
        continue

    server_url = f.get("ServerRelativeUrl", "")
    size = int(f.get("Length", 0))
    out_path = os.path.join(OUT_DIR, name)

    if os.path.exists(out_path) and os.path.getsize(out_path) == size:
        print(f"  {name}: already downloaded, skipping.")
        continue

    # Download using the SharePoint download endpoint
    download_url = f"{BASE}{server_url}"
    print(f"  Downloading {name} ({size / 1024 / 1024:.1f} MB)...")

    resp_dl = session.get(download_url, stream=True, headers={"Accept": "application/octet-stream"})
    if resp_dl.status_code != 200:
        # Try the _layouts/download.aspx approach
        download_url2 = f"{BASE}/personal/pguo4_jh_edu/_layouts/15/download.aspx?SourceUrl={urllib.parse.quote(server_url)}"
        print(f"    Direct failed ({resp_dl.status_code}), trying download.aspx...")
        resp_dl = session.get(download_url2, stream=True)

    if resp_dl.status_code == 200:
        downloaded = 0
        with open(out_path, "wb") as out_f:
            for chunk in resp_dl.iter_content(chunk_size=1024 * 1024):
                out_f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / size * 100 if size else 0
                print(f"    {downloaded / 1024 / 1024:.1f} / {size / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="\r")
        print(f"\n    Saved to {out_path} ({os.path.getsize(out_path)} bytes)")
    else:
        print(f"    Download failed: HTTP {resp_dl.status_code}")
        print(f"    Response: {resp_dl.text[:300]}")

# Also copy the X4 as the default reconformer_checkpoint.pth
x4_path = os.path.join(OUT_DIR, "F_X4_checkpoint.pth")
default_path = os.path.join(OUT_DIR, "reconformer_checkpoint.pth")
if os.path.exists(x4_path) and not os.path.exists(default_path):
    import shutil
    shutil.copy2(x4_path, default_path)
    print(f"\nCopied F_X4_checkpoint.pth -> reconformer_checkpoint.pth")

print("\nDone!")
