# To generate documentation htmls:

* Install the prerequisites:

```shell
conda install -c conda-forge pandoc
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

* Refer to [workaround](workaround_passpipeline_html.sh) if the notebook
  documents with run results are to be demo on websites.

* To check the generated htmls:

```shell
cd build/html
python -m http.server 8080
```

* Deploy to Github Page


TODO: Move the flow to ReadtheDocs if it's not that necessary to demo notebook
docs with run results. Revisit this in future.
