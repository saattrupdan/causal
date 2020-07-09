import pyro
import pyro.distributions as dist

def weather():
    cloudy = pyro.sample('cloudy', dist.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1. else 'sunny'
    mean_temp = {'cloudy': 55., 'sunny': 75.}[cloudy]
    scale_temp = {'cloudy': 10., 'sunny': 15.}[cloudy]
    temp = pyro.sample('temp', dist.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80. else 50.
    ice_cream = pyro.sample('ice_cream', dist.Normal(expected_sales, 10.))
    return ice_cream.item()

if __name__ == '__main__':
    print('Weather samples:')
    for _ in range(3):
        print(weather())

    print('Ice cream sales samples:')
    for _ in range(3):
        print(ice_cream_sales())
