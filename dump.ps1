$root = (Get-Location).Path
$output = Join-Path $root "dump.txt"

# Dossiers à ignorer (regex)
$excludePathRegex = "\\(__pycache__|venv|\.venv|\.git|node_modules)\\"

# Vide/crée le dump
"" | Set-Content -Path $output -Encoding utf8

Get-ChildItem -Path $root -Recurse -File -Filter "*.py" |
  Where-Object { $_.FullName -notmatch $excludePathRegex } |
  Sort-Object FullName |
  ForEach-Object {
    Add-Content -Path $output -Encoding utf8 -Value ("`n===== FILE: {0} =====`n" -f $_.FullName)
    try {
      $content = Get-Content -Path $_.FullName -Raw -ErrorAction Stop
      Add-Content -Path $output -Encoding utf8 -Value $content
    } catch {
      Add-Content -Path $output -Encoding utf8 -Value ("[ERROR reading file] {0}" -f $_.Exception.Message)
    }
    Add-Content -Path $output -Encoding utf8 -Value "`n`n"
  }

Write-Host "OK -> $output"
