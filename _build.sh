#!/bin/sh

Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

git config --global user.email "yufree@live.cn"
git config --global user.name "Yufree"
