#!/bin/bash

msg=$1

# 1. 添加
git add .

# 2. 提交
git commit -m "$msg"

# 3. 拉取并rebase（关键）
git pull --rebase origin main

# 4. 推送
git push origin main