#!/bin/bash

# Git LFS Large Files Cleanup Script
# This script sets up Git LFS and removes large files from Git history

echo "🚀 Starting Git LFS cleanup process..."

# Step 1: Initialize Git LFS if not already done
echo "📦 Installing Git LFS..."
git lfs install

# Step 2: Configure .gitattributes for large file types
echo "⚙️ Configuring .gitattributes..."
cat > .gitattributes << 'EOF'
* text=auto eol=lf
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ts filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
EOF

# Step 3: If repository has existing large files in history, clean them
if [ $(git rev-list --count HEAD 2>/dev/null || echo 0) -gt 0 ]; then
    echo "🧹 Cleaning existing Git history..."
    
    # Check if git-filter-repo is available
    if command -v git-filter-repo &> /dev/null; then
        echo "Using git-filter-repo for history cleanup..."
        git filter-repo --strip-blobs-bigger-than 100M --force
    else
        echo "⚠️ git-filter-repo not found. Installing..."
        pip install git-filter-repo
        
        if command -v git-filter-repo &> /dev/null; then
            git filter-repo --strip-blobs-bigger-than 100M --force
        else
            echo "❌ Could not install git-filter-repo. Manual cleanup required."
            echo "Please install git-filter-repo: pip install git-filter-repo"
            exit 1
        fi
    fi
fi

# Step 4: Re-add large files via LFS
echo "📁 Re-adding large files via Git LFS..."
git add .gitattributes
git add models/ 2>/dev/null || true

# Step 5: Commit cleaned repository
echo "💾 Committing cleaned repository..."
git commit -m "Clean large files from Git history and push via Git LFS"

# Step 6: Show repository status
echo "📊 Repository Status:"
echo "LFS tracked files:"
git lfs ls-files

echo -e "\nRepository size:"
git count-objects -vH

echo -e "\n✅ Cleanup complete! Repository is ready to push to GitHub."
echo "To push to GitHub, run: git remote add origin <your-repo-url> && git push -u origin master"
