#!/bin/sh

Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

git config --global user.email "yufree@live.cn"
git config --global user.name "Yufree"

# clone the repository to the book-output directory
git clone -b gh-pages \
  https://github.com/yufree/metaworkflow.git \
  book-output
cd book-output
cp -r ../_book/* ./
git add --all *
git commit -m"Update the book"
git push origin gh-pages