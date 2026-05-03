Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(scriptDir, "..\..\..\.venv\Scripts\pythonw.exe")
pythonw = fso.GetAbsolutePathName(pythonw)
gui = fso.BuildPath(scriptDir, "gui.py")

shell.CurrentDirectory = scriptDir
shell.Run """" & pythonw & """ """ & gui & """", 0, False
