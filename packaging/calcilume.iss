#ifndef AppVersion
  #error AppVersion is required
#endif
[Setup]
AppId={{528B9804-6026-4D14-B201-BB87DBC22234}
AppName=CalciLume 钙荧
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\CalciLume
DefaultGroupName=CalciLume
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=calcilume.ico
UninstallDisplayIcon={app}\runtime\MiniscopeAnalysis.exe
OutputDir=..\dist
OutputBaseFilename=CalciLume-Windows-Setup
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
[Files]
Source: "..\dist\MiniscopeAnalysis\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked
[Icons]
Name: "{autoprograms}\CalciLume"; Filename: "{app}\runtime\MiniscopeAnalysis.exe"
Name: "{autodesktop}\CalciLume"; Filename: "{app}\runtime\MiniscopeAnalysis.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\runtime\MiniscopeAnalysis.exe"; Description: "{cm:LaunchProgram,CalciLume}"; Flags: nowait postinstall skipifsilent
