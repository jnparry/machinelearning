from crossVal import cross_val
from load_mpg import load_mpg
from load_aut import load_aut
from load_car import load_car
from predictWithKNN import knn_predict
from predictWithRegression import regression_predict

k = 5

car_data, car_target = load_car()
knn_predict(car_data, car_target, k)
cross_val(car_data, car_target, k)

aut_data, aut_target = load_aut()
knn_predict(aut_data, aut_target, k)
cross_val(aut_data, aut_target, k)

# the data is all the other attributes, the target is the mpg
mpg_data, mpg_target = load_mpg()
regression_predict(mpg_data, mpg_target, k)
cross_val(mpg_data, mpg_target, k)

