# Hexo Blog Guide

This is a Hexo blog deployed to GitHub Pages. The repo has two branches:

- **`hexo`** — source files (markdown posts, config, themes). Work here.
- **`master`** — generated static site. Don't edit directly; it's overwritten by deploy.

## Setup (after cloning or first time)

```bash
git checkout hexo
npm install
```

The NexT theme is in `themes/next/` (cloned from theme-next/hexo-theme-next v7.8.0).

**Node version**: Tested with Node 25. hexo 5.4.2 is in `package.json`.

## Daily Workflow

### 1. Write a new post

```bash
./node_modules/.bin/hexo new "My Post Title"
# Creates: source/_posts/My-Post-Title.md
```

Edit the file. Front matter example:

```yaml
---
title: My Post Title
date: 2026-05-06 12:00:00
tags:
  - Tag1
  - Tag2
---

Post content here...
```

### 2. Preview locally

```bash
./node_modules/.bin/hexo server
# Open http://localhost:4000
```

### 3. Generate the static site

```bash
./node_modules/.bin/hexo generate
# Output goes to public/
```

### 4. Deploy to GitHub Pages (master branch)

```bash
./node_modules/.bin/hexo deploy
# Pushes public/ contents to origin/master
```

Or generate + deploy in one step:

```bash
./node_modules/.bin/hexo deploy --generate
```

### 5. Commit your source changes

```bash
git add .
git commit -m "Add new post: My Post Title"
git push origin hexo
```

## Adding a New Page (not a blog post)

```bash
./node_modules/.bin/hexo new page "page-name"
# Creates: source/page-name/index.md
```

Or create the file manually:

```
source/superteam/index.md  →  https://crysple.github.io/superteam/
```

## Useful Commands

| Command | Description |
|---------|-------------|
| `hexo new "Title"` | Create new post |
| `hexo new page "name"` | Create new page |
| `hexo server` | Local preview at localhost:4000 |
| `hexo generate` | Build static site to `public/` |
| `hexo deploy` | Push `public/` to master branch |
| `hexo clean` | Clear cache and public folder |
| `hexo list post` | List all posts |

## Configuration

Main config: `_config.yml`
- `title`, `subtitle`, `author` — site identity
- `url` — set to `https://crysple.github.io` for proper links
- `theme: next` — uses NexT theme in `themes/next/`
- `deploy.repo` — GitHub repo SSH URL
- `deploy.branch: master` — deploys to master branch

NexT theme config: `themes/next/_config.yml`

## Known Issues / Fixes Applied

- **hexo 3.9.0 + Node 25**: `util.isDate` removed. Updated to hexo 5.4.2.
- **LaTeX `{{ }}` conflicts with nunjucks**: Fixed by setting `marked.disableNunjucks: true` in `_config.yml`.
- **hexo-inject**: Removed (caused empty HTML files with NexT 7.8.0).
- **NexT theme**: Must be in `themes/next/` (not `themes/hexo-theme-next/`) to match `theme: next` in config.
