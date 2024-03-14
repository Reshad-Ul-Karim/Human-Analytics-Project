from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Initialize GoogleAuth and load client secrets


if not gauth.credentials:
    gauth.LocalWebserverAuth()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile("client_secrets.json")
drive = GoogleDrive(gauth)

# Replace 'your_folder_id_here' with the actual folder ID from Google Drive
file_list = drive.ListFile({'q': f"'My Drive' in parents and trashed=false"}).GetList()
for file in file_list:
    print('title: %s, id: %s' % (file['title'], file['id']))
