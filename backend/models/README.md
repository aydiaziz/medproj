Place the MediaPipe Task model file `hand_landmarker.task` in this directory.

How to get the model
- Download the official MediaPipe Hand Landmarker task model from MediaPipe's model releases or cloud storage. The exact URL depends on the model version.

Quick steps (PowerShell)
1. From project root run (replace URL with actual model URL):

   .\backend\download_model.ps1 -Url "https://.../hand_landmarker.task"

2. Verify the file exists:

   Test-Path .\backend\models\hand_landmarker.task

If you prefer, provide the model URL and I can download it for you.