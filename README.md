# Streamlit App: churn classification

## Prerequisites

- Pipenv

## Install dependencies

```bash
pipenv shell
pipenv install
```



## Run the app locally

```bash
streamlit run app.py
# the app will be accessiable at http://localhost:8501/
```

## Deploy the app to heroku

1. Create account in heroku.com
2. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli#download-and-install
3. Create new app
``` bash
heroku create <MY_HEROKU_APP>
```
4. Add `heroku` repository: 

``` bash
heroku git:remote -a <MY_HEROKU_APP>
```


5. Deploy to Heroku

```bash
git push heroku HEAD:master
```

The app will be accessible at https://<MY_HEROKU_APP>.herokuapp.com/ 

e.g. https://streamlit-app-churn.herokuapp.com/
