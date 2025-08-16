# Git LFS Large Files Cleanup Script
# This script sets up Git LFS and removes large files from Git history

Write-Host "🚀 Starting Git LFS cleanup process..." -ForegroundColor Green

# Step 1: Initialize Git LFS if not already done
Write-Host "📦 Installing Git LFS..." -ForegroundColor Yellow
git lfs install

# Step 2: Configure .gitattributes for large file types
Write-Host "⚙️ Configuring .gitattributes..." -ForegroundColor Yellow
@"
* text=auto eol=lf
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ts filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
"@ | Out-File -FilePath ".gitattributes" -Encoding UTF8

# Step 3: If repository has existing large files in history, clean them
if (git rev-list --count HEAD -gt 0) {
    Write-Host "🧹 Cleaning existing Git history..." -ForegroundColor Yellow
    
    # Check if git-filter-repo is available
    try {
        git filter-repo --version | Out-Null
        Write-Host "Using git-filter-repo for history cleanup..." -ForegroundColor Blue
        
        # Remove large files from history
        git filter-repo --strip-blobs-bigger-than 100M --force
        
    } catch {
        Write-Host "⚠️ git-filter-repo not found. Using alternative method..." -ForegroundColor Red
        Write-Host "Installing git-filter-repo via pip..." -ForegroundColor Yellow
        
        try {
            pip install git-filter-repo
            git filter-repo --strip-blobs-bigger-than 100M --force
        } catch {
            Write-Host "❌ Could not install git-filter-repo. Manual cleanup required." -ForegroundColor Red
            Write-Host "Please install git-filter-repo: pip install git-filter-repo" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Step 4: Re-add large files via LFS
Write-Host "📁 Re-adding large files via Git LFS..." -ForegroundColor Yellow
git add .gitattributes
git add models/

# Step 5: Commit cleaned repository
Write-Host "💾 Committing cleaned repository..." -ForegroundColor Yellow
git commit -m "Clean large files from Git history and push via Git LFS"

# Step 6: Show repository status
Write-Host "📊 Repository Status:" -ForegroundColor Green
Write-Host "LFS tracked files:" -ForegroundColor Cyan
git lfs ls-files

Write-Host "`nRepository size:" -ForegroundColor Cyan
git count-objects -vH

Write-Host "`n✅ Cleanup complete! Repository is ready to push to GitHub." -ForegroundColor Green
Write-Host "To push to GitHub, run: git remote add origin <your-repo-url> && git push -u origin master" -ForegroundColor Yellow
