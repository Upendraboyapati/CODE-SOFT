{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "5b7a6479-448c-4f98-80ba-47aef755aeeb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>moviename</th>\n",
       "      <th>genre</th>\n",
       "      <th>plot</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Oscar et la dame rose (2009)</td>\n",
       "      <td>drama</td>\n",
       "      <td>Listening in to a conversation between his do...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Cupid (1997)</td>\n",
       "      <td>thriller</td>\n",
       "      <td>A brother and sister with a past incestuous r...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Young, Wild and Wonderful (1980)</td>\n",
       "      <td>adult</td>\n",
       "      <td>As the bus empties the students for their fie...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>The Secret Sin (1915)</td>\n",
       "      <td>drama</td>\n",
       "      <td>To help their unemployed father make ends mee...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>The Unrecovered (2007)</td>\n",
       "      <td>drama</td>\n",
       "      <td>The film's title refers not only to the un-re...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                           moviename       genre  \\\n",
       "0   1       Oscar et la dame rose (2009)       drama    \n",
       "1   2                       Cupid (1997)    thriller    \n",
       "2   3   Young, Wild and Wonderful (1980)       adult    \n",
       "3   4              The Secret Sin (1915)       drama    \n",
       "4   5             The Unrecovered (2007)       drama    \n",
       "\n",
       "                                                plot  \n",
       "0   Listening in to a conversation between his do...  \n",
       "1   A brother and sister with a past incestuous r...  \n",
       "2   As the bus empties the students for their fie...  \n",
       "3   To help their unemployed father make ends mee...  \n",
       "4   The film's title refers not only to the un-re...  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#first we need to impot allthe required libraries,modules \n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "#naming the features from description data set\n",
    "trainset=pd.read_csv(\"train_data.csv\",delimiter=\":::\",engine=\"python\",header=None,names=['id','moviename','genre','plot'])\n",
    "trainset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "38523e3f-cfad-4e3b-9969-abff18fd6aaa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(54214, 4)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "c1a1bf16-f412-4850-ab83-baea21995454",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 54214 entries, 0 to 54213\n",
      "Data columns (total 4 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   id         54214 non-null  int64 \n",
      " 1   moviename  54214 non-null  object\n",
      " 2   genre      54214 non-null  object\n",
      " 3   plot       54214 non-null  object\n",
      "dtypes: int64(1), object(3)\n",
      "memory usage: 1.7+ MB\n"
     ]
    }
   ],
   "source": [
    "trainset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "82e41f6c-ecc8-4b10-847d-c2edda652c27",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>54214.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>27107.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>15650.378084</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>13554.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>27107.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>40660.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>54214.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 id\n",
       "count  54214.000000\n",
       "mean   27107.500000\n",
       "std    15650.378084\n",
       "min        1.000000\n",
       "25%    13554.250000\n",
       "50%    27107.500000\n",
       "75%    40660.750000\n",
       "max    54214.000000"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "246c18bb-72da-4a02-b741-d4b75e3844c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>moviename</th>\n",
       "      <th>plot</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Edgar's Lunch (1998)</td>\n",
       "      <td>L.R. Brane loves his life - his car, his apar...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>La guerra de papá (1977)</td>\n",
       "      <td>Spain, March 1964: Quico is a very naughty ch...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Off the Beaten Track (2010)</td>\n",
       "      <td>One year in the life of Albin and his family ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Meu Amigo Hindu (2015)</td>\n",
       "      <td>His father has died, he hasn't spoken with hi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Er nu zhai (1955)</td>\n",
       "      <td>Before he was known internationally as a mart...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                      moviename  \\\n",
       "0   1          Edgar's Lunch (1998)    \n",
       "1   2      La guerra de papá (1977)    \n",
       "2   3   Off the Beaten Track (2010)    \n",
       "3   4        Meu Amigo Hindu (2015)    \n",
       "4   5             Er nu zhai (1955)    \n",
       "\n",
       "                                                plot  \n",
       "0   L.R. Brane loves his life - his car, his apar...  \n",
       "1   Spain, March 1964: Quico is a very naughty ch...  \n",
       "2   One year in the life of Albin and his family ...  \n",
       "3   His father has died, he hasn't spoken with hi...  \n",
       "4   Before he was known internationally as a mart...  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "testset=pd.read_csv(\"test_data.csv\",delimiter=\":::\",engine=\"python\",header=None,names=['id','moviename','plot'])\n",
    "testset.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "3479886d-c9d0-4b79-9640-6f19623895d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>moviename</th>\n",
       "      <th>plot</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Edgar's Lunch (1998)</td>\n",
       "      <td>L.R. Brane loves his life - his car, his apar...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>La guerra de papá (1977)</td>\n",
       "      <td>Spain, March 1964: Quico is a very naughty ch...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Off the Beaten Track (2010)</td>\n",
       "      <td>One year in the life of Albin and his family ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Meu Amigo Hindu (2015)</td>\n",
       "      <td>His father has died, he hasn't spoken with hi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Er nu zhai (1955)</td>\n",
       "      <td>Before he was known internationally as a mart...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                      moviename  \\\n",
       "0   1          Edgar's Lunch (1998)    \n",
       "1   2      La guerra de papá (1977)    \n",
       "2   3   Off the Beaten Track (2010)    \n",
       "3   4        Meu Amigo Hindu (2015)    \n",
       "4   5             Er nu zhai (1955)    \n",
       "\n",
       "                                                plot  \n",
       "0   L.R. Brane loves his life - his car, his apar...  \n",
       "1   Spain, March 1964: Quico is a very naughty ch...  \n",
       "2   One year in the life of Albin and his family ...  \n",
       "3   His father has died, he hasn't spoken with hi...  \n",
       "4   Before he was known internationally as a mart...  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "testsetsolution=pd.read_csv(\"test_data.csv\",delimiter=\":::\",engine=\"python\",header=None,names=['id','moviename','plot'])\n",
    "testsetsolution.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "53109bac-36c7-4280-b91c-c468caea6889",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract features using tf-idf vectorizer \n",
    "vectorizer = TfidfVectorizer(max_features=5000)\n",
    "X_train = vectorizer.fit_transform(trainset['plot'])\n",
    "y_train = trainset['genre']\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b70af65-db15-41a5-97cb-63ae6ec84d12",
   "metadata": {},
   "outputs": [],
   "source": [
    "# traing the model by logistic regression by correlating datasets\n",
    "classifier = LogisticRegression(max_iter=1000)\n",
    "classifier.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3a0e799-96a3-49fa-8301-1dd70db36be8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transform test set plots into TF-IDF vectors \n",
    "X_test = vectorizer.transform(testset['plot'])\n",
    "\n",
    "# Predict genres for the test set plots \n",
    "predicted_genres = classifier.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "2f4876c8-6d81-47fa-a0cb-ee8aa0581018",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[' drama ' ' drama ' ' documentary ' ... ' comedy ' ' drama '\n",
      " ' documentary ']\n"
     ]
    }
   ],
   "source": [
    "#predictions that i got\n",
    "print(predicted_genres)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "557e71a7-13cb-4150-b4db-265a7726b5e8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
