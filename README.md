# Covid-19 Compartmental Simple Model
This repository is a simple realisation of mathematical model of infectious disease Covid-19. 

Main idea of model is solving differential equations: [The SIR model without vital dynamics](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)

Source of data : [github repository](https://github.com/CSSEGISandData/COVID-19)

Competition : [ODS.ai](https://ods.ai/competitions/sberbank-covid19-forecast)

## Installation

Clone repository in folder on your system.
```bash
git clone https://github.com/lasados/Covid19.git
```
Open repository.

Install requirements with pip.

```bash
pip install -r pip-requirements.txt
```

## Usage
```python
import sys
sys.path.append('../')

from models.covidprocess import DataCovid
from models.linear import NaiveLinearModel
from models.compart import SIR


data = DataCovid().read()
linear_model = NaiveLinearModel(data)
linear_model.plot(start_fit=60)

model = SIR(data)
model.plot()
```

See more details in docs of modules [here...](https://github.com/lasados/Covid19/tree/master/models)

See also example [here...](https://github.com/lasados/Covid19/tree/master/notebooks)

