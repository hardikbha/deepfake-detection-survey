#!/bin/bash
# GitHub Repository Setup Script for Deepfake Detection Survey
# Run this script to initialize and push to GitHub

echo "🚀 Deepfake Detection Survey - GitHub Setup Script"
echo "==================================================="

# Configuration - UPDATE THESE VALUES
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
REPO_NAME="deepfake-detection-survey"
REPO_DESCRIPTION="Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets"

echo ""
echo "📋 Prerequisites:"
echo "   1. Install Git: https://git-scm.com/"
echo "   2. Create GitHub account: https://github.com/"
echo "   3. Generate Personal Access Token: https://github.com/settings/tokens"
echo "   4. Install GitHub CLI (optional): brew install gh"
echo ""

# Step 1: Initialize git repository
echo "Step 1: Initializing Git repository..."
cd "$(dirname "$0")"
git init

# Step 2: Configure git (if not already done)
echo ""
echo "Step 2: Configure Git (if needed)..."
echo "   Run these commands if not configured:"
echo "   git config --global user.name \"Your Name\""
echo "   git config --global user.email \"your.email@example.com\""

# Step 3: Add all files
echo ""
echo "Step 3: Adding files to staging..."
git add .

# Step 4: Create initial commit
echo ""
echo "Step 4: Creating initial commit..."
git commit -m "Initial commit: Deepfake Detection Survey repository

- Added comprehensive README with detection literature
- Added datasets documentation and download script
- Added implementations reference
- Added benchmark results structure
- Added citation information"

# Step 5: Create GitHub repository (using GitHub CLI)
echo ""
echo "Step 5: Creating GitHub repository..."
echo ""
echo "Option A (GitHub CLI):"
echo "   gh repo create $REPO_NAME --public --description \"$REPO_DESCRIPTION\" --source=. --remote=origin --push"
echo ""
echo "Option B (Manual):"
echo "   1. Go to https://github.com/new"
echo "   2. Repository name: $REPO_NAME"
echo "   3. Description: $REPO_DESCRIPTION"
echo "   4. Set to Public"
echo "   5. Do NOT initialize with README (we already have one)"
echo "   6. Click 'Create repository'"
echo ""

# Step 6: Add remote and push
echo "Step 6: After creating the repo on GitHub, run:"
echo "   git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"

echo ""
echo "✅ Setup script complete!"
echo ""
echo "📝 Quick Commands Reference:"
echo "   git status              # Check status"
echo "   git add .               # Stage all changes"
echo "   git commit -m \"message\" # Commit changes"
echo "   git push                # Push to GitHub"
