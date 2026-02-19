$base = "D:\onedrive\startup\program\physics_world_model\PWM4\Physics_World_Model-master\papers"
$outDir = "$base\arxiv_submissions"

# --- Rail Paper ZIP ---
$railTemp = "$outDir\rail_temp"
Remove-Item -ErrorAction SilentlyContinue "$outDir\rail_paper_arxiv.zip"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $railTemp
New-Item -ItemType Directory -Force $railTemp | Out-Null
New-Item -ItemType Directory -Force "$railTemp\sections" | Out-Null

Copy-Item "$base\rail\main.tex" $railTemp
Copy-Item "$base\rail\preamble.tex" $railTemp
Copy-Item "$base\rail\main.bbl" $railTemp
Copy-Item "$base\rail\rail_paper.bib" $railTemp
Copy-Item "$base\rail\sections\*.tex" "$railTemp\sections\"

Compress-Archive -Path "$railTemp\*" -DestinationPath "$outDir\rail_paper_arxiv.zip"
Remove-Item -Recurse -Force $railTemp
Write-Host "Rail Paper ZIP created: $outDir\rail_paper_arxiv.zip"

# --- Flagship Paper ZIP ---
$flagTemp = "$outDir\flagship_temp"
Remove-Item -ErrorAction SilentlyContinue "$outDir\flagship_paper_arxiv.zip"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $flagTemp
New-Item -ItemType Directory -Force $flagTemp | Out-Null

Copy-Item "$base\pwm_flagship\main.tex" $flagTemp
Copy-Item "$base\pwm_flagship\preamble.tex" $flagTemp
Copy-Item "$base\pwm_flagship\methods.tex" $flagTemp
Copy-Item "$base\pwm_flagship\supplementary.tex" $flagTemp
Copy-Item "$base\pwm_flagship\main.bbl" $flagTemp
Copy-Item "$base\pwm_flagship\pwm_flagship.bib" $flagTemp
# Include naturemag.bst since arXiv may not have it
Copy-Item "C:\Users\integ\AppData\Local\Programs\MiKTeX\bibtex\bst\nature\naturemag.bst" $flagTemp

Compress-Archive -Path "$flagTemp\*" -DestinationPath "$outDir\flagship_paper_arxiv.zip"
Remove-Item -Recurse -Force $flagTemp
Write-Host "Flagship Paper ZIP created: $outDir\flagship_paper_arxiv.zip"
