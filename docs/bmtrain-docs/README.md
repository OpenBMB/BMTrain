建立本地文档

可以通过 sphinx 来建立本地文档，步骤如下。 

```
pip install sphinx
pip install sphinx_copybutton
pip install myst_parser
pip install recommonmark
pip install sphinx-markdown-tables
pip install sphinx_rtd_theme
pip install sphinx_copybutton
pip install sphinx_toolbox
```

然后:
```
cd bmtrain-docs
make html
```
最后，用浏览器打开`build/html`, 即可本地查看文档。 
