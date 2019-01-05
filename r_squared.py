from statistics import mean
import numpy as np

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    b=mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys,regression_line):
    y_mean=[mean(ys) for y in ys]
    squared_error_regr = squared_error(ys,regression_line)
    squared_error_y_mean = squared_error(ys,y_mean)
    return 1 -(squared_error_regr/squared_error_y_mean)


m,b=best_fit_slope(xs,ys)
print(m,b)
regression_line=[(m*x) +b for x in xs]
print(type(regression_line))
print("data in the regression line is ",regression_line)

predict_x=7
predict_y=(m*predict_x)+b


r_squared = coefficient_of_determination(ys,regression_line)

print("the r_squared is ",r_squared)
