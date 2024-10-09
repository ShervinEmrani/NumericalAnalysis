import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# function used for cubic spline
'''
    def __init__(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        n = len(x)

        bc, y = self._validate_bc(bc_type, y, y.shape[1:], axis)

        if extrapolate is None:
            if bc[0] == 'periodic':
                extrapolate = 'periodic'
            else:
                extrapolate = True

        if y.size == 0:
            # bail out early for zero-sized arrays
            s = np.zeros_like(y)
        else:
            dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
            slope = np.diff(y, axis=0) / dxr

            # If bc is 'not-a-knot' this change is just a convention.
            # If bc is 'periodic' then we already checked that y[0] == y[-1],
            # and the spline is just a constant, we handle this case in the
            # same way by setting the first derivatives to slope, which is 0.
            if n == 2:
                if bc[0] in ['not-a-knot', 'periodic']:
                    bc[0] = (1, slope[0])
                if bc[1] in ['not-a-knot', 'periodic']:
                    bc[1] = (1, slope[0])

            # This is a special case, when both conditions are 'not-a-knot'
            # and n == 3. In this case 'not-a-knot' can't be handled regularly
            # as the both conditions are identical. We handle this case by
            # constructing a parabola passing through given points.
            if n == 3 and bc[0] == 'not-a-knot' and bc[1] == 'not-a-knot':
                A = np.zeros((3, 3))  # This is a standard matrix.
                b = np.empty((3,) + y.shape[1:], dtype=y.dtype)

                A[0, 0] = 1
                A[0, 1] = 1
                A[1, 0] = dx[1]
                A[1, 1] = 2 * (dx[0] + dx[1])
                A[1, 2] = dx[0]
                A[2, 1] = 1
                A[2, 2] = 1

                b[0] = 2 * slope[0]
                b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
                b[2] = 2 * slope[1]

                s = solve(A, b, overwrite_a=True, overwrite_b=True,
                          check_finite=False)
            elif n == 3 and bc[0] == 'periodic':
                # In case when number of points is 3 we compute the derivatives
                # manually
                s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
                t = (slope / dxr).sum() / (1. / dxr).sum()
                s.fill(t)
            else:
                # Find derivative values at each x[i] by solving a tridiagonal
                # system.
                A = np.zeros((3, n))  # This is a banded matrix representation.
                b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

                # Filling the system for i=1..n-2
                #                         (x[i-1] - x[i]) * s[i-1] +\
                # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
                #                         (x[i] - x[i-1]) * s[i+1] =\
                #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
                #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

                A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
                A[0, 2:] = dx[:-1]                   # The upper diagonal
                A[-1, :-2] = dx[1:]                  # The lower diagonal

                b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

                bc_start, bc_end = bc

                if bc_start == 'periodic':
                    # Due to the periodicity, and because y[-1] = y[0], the
                    # linear system has (n-1) unknowns/equations instead of n:
                    A = A[:, 0:-1]
                    A[1, 0] = 2 * (dx[-1] + dx[0])
                    A[0, 1] = dx[-1]

                    b = b[:-1]

                    # Also, due to the periodicity, the system is not tri-diagonal.
                    # We need to compute a "condensed" matrix of shape (n-2, n-2).
                    # See https://web.archive.org/web/20151220180652/http://www.cfm.brown.edu/people/gk/chap6/node14.html
                    # for more explanations.
                    # The condensed matrix is obtained by removing the last column
                    # and last row of the (n-1, n-1) system matrix. The removed
                    # values are saved in scalar variables with the (n-1, n-1)
                    # system matrix indices forming their names:
                    a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
                    a_m1_m2 = dx[-1]
                    a_m1_m1 = 2 * (dx[-1] + dx[-2])
                    a_m2_m1 = dx[-3]
                    a_0_m1 = dx[0]

                    b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
                    b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])

                    Ac = A[:, :-1]
                    b1 = b[:-1]
                    b2 = np.zeros_like(b1)
                    b2[0] = -a_0_m1
                    b2[-1] = -a_m2_m1

                    # s1 and s2 are the solutions of (n-2, n-2) system
                    s1 = solve_banded((1, 1), Ac, b1, overwrite_ab=False,
                                      overwrite_b=False, check_finite=False)

                    s2 = solve_banded((1, 1), Ac, b2, overwrite_ab=False,
                                      overwrite_b=False, check_finite=False)

                    # computing the s[n-2] solution:
                    s_m1 = ((b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1]) /
                            (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1]))

                    # s is the solution of the (n, n) system:
                    s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
                    s[:-2] = s1 + s_m1 * s2
                    s[-2] = s_m1
                    s[-1] = s[0]
                else:
                    if bc_start == 'not-a-knot':
                        A[1, 0] = dx[1]
                        A[0, 1] = x[2] - x[0]
                        d = x[2] - x[0]
                        b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
                                dxr[0]**2 * slope[1]) / d
                    elif bc_start[0] == 1:
                        A[1, 0] = 1
                        A[0, 1] = 0
                        b[0] = bc_start[1]
                    elif bc_start[0] == 2:
                        A[1, 0] = 2 * dx[0]
                        A[0, 1] = dx[0]
                        b[0] = -0.5 * bc_start[1] * dx[0]**2 + 3 * (y[1] - y[0])

                    if bc_end == 'not-a-knot':
                        A[1, -1] = dx[-2]
                        A[-1, -2] = x[-1] - x[-3]
                        d = x[-1] - x[-3]
                        b[-1] = ((dxr[-1]**2*slope[-2] +
                                 (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)
                    elif bc_end[0] == 1:
                        A[1, -1] = 1
                        A[-1, -2] = 0
                        b[-1] = bc_end[1]
                    elif bc_end[0] == 2:
                        A[1, -1] = 2 * dx[-1]
                        A[-1, -2] = dx[-1]
                        b[-1] = 0.5 * bc_end[1] * dx[-1]**2 + 3 * (y[-1] - y[-2])

                    s = solve_banded((1, 1), A, b, overwrite_ab=True,
                                     overwrite_b=True, check_finite=False)

'''


# f'(x) = (f(x+h)-f(x))/h
def derivative(function, x, h):
    upper_division = function(x+h)-function(x)
    lower_division = h
    return upper_division/lower_division


# Read file
Male = pd.read_excel("./Male.xlsx")

# cleaning data
# select rows
Male_modified = Male[16:]
# select columns
Male_modified_columns = Male.iloc[15].values.tolist()
Male_modified.columns = Male_modified_columns

# Removing redundant info
Male_modified = Male_modified.reset_index().drop(columns=["index", "Index", "Variant", "Notes", "Location code", "ISO3 Alpha-code", "ISO2 Alpha-code", "SDMX code**", "Type", "Parent code"])
# Renaming the column name to Region
Male_modified = Male_modified.rename(columns={'Region, subregion, country or area *': 'Region'})

# Set World to 0 and Iran to 1
Male_modified.loc[Male_modified['Region'] == "WORLD", 'Region'] = 0
Male_modified.loc[Male_modified['Region'] == "Iran (Islamic Republic of)", 'Region'] = 1

print(Male_modified)

print(3)

# changing [1, 2012, 0, 1, 2, ..., 100] to [[1, 2012, 0], [1, 2012, 1], ..., [1, 2012, 100]] (Ti transform)
new_values = []
# labels
labels = []

# select rows from table
for rows in range(144):
    # select region
    region = Male_modified.iloc[rows].values[0]
    # select year
    year = Male_modified.iloc[rows].values[1]
    # find population based on year and region
    population = Male_modified.iloc[rows].values[2:]
    # append to labels
    labels.append(list(population))
    # population of all ages
    original_list = np.arange(101)
    # Ti transform
    new_values += [[region, year, i] for i in original_list]
# set the new_list as numpy array
new_list = np.array(new_values)
# reshape the new_list for labels
new_list_labels = np.array(labels).reshape(-1, 1)

# Interpolation by Age
selected_year = 143
print("(", Male_modified.loc[selected_year]["Region"], ",", Male_modified.loc[selected_year]["Year"], ")")
cs = CubicSpline(new_list[101*selected_year:101*(selected_year+1), 2], new_list_labels.reshape(1, -1)[0][101*selected_year:101*(selected_year+1)])
print("derivative for 40 year olds: ", (cs(40+1/365)-cs(40))/(1/365))

y_pred_sorted = []
space = np.linspace(0, 100, num=100)
for i in space:
    y_pred_sorted.append(cs(i))

plt.figure(figsize=(10, 6))
plt.plot(space, y_pred_sorted, 'b.-.', label='Interpolated')
plt.xlabel('Age')
plt.ylabel('Population')
plt.title('Spline Graph')
plt.legend()
plt.show()

#  اختلاف نرخ سالخوردگی 40 ساله ها در مردان در ایران بین سال های 2002 و 2000
# طول گام مشتق = 1 روز
# selects year 2000
selected_row = 122
Iran2000 = CubicSpline(new_list[101*selected_row:101*(selected_row+1), 2], new_list_labels.reshape(1, -1)[0][101*selected_row:101*(selected_row+1)])
# selects year 2002
selected_row = 124
Iran2002 = CubicSpline(new_list[101*selected_row:101*(selected_row+1), 2], new_list_labels.reshape(1,-1)[0][101*selected_row:101*(selected_row+1)])

print("ِDifference : ", derivative(Iran2002, 40, 1/365)-derivative(Iran2000, 40, 1/365))


# در کدام سال در کشور ایران در بین مردان اختلاف میانگین نرخ سالخوردگی جمعیت زیر 50 ساله ها نسبت به بالای 50 ساله ها بیشینه بود؟ 
listOfDifferences = []
# selects years in Iran
for select in range(72, 144):
    IranYears = CubicSpline(new_list[101*select:101*(select+1), 2], new_list_labels.reshape(1, -1)[0][101*select:101*(select+1)])
    listOfUnder50Derivatives = []
    for i in range(51):
        listOfUnder50Derivatives.append(derivative(IranYears, i, 1/365))
    listOfOver50Derivatives = []
    for i in range(51, 101):
        listOfOver50Derivatives.append(derivative(IranYears, i, 1/365))
    difference = np.mean(np.array(listOfOver50Derivatives))-np.mean(np.array(listOfUnder50Derivatives))
    listOfDifferences.append(difference)
print(np.argmin(np.array(listOfDifferences))+1950)

# در کدام سال ها میانگین نرخ سالخوردگی جمعیت 13 تا 19 ساله ها در مردان در ایران بیشتر از همین گروه سنی در کل جهان بود
meanOfIraniansDerivatives = []
for select in range(72):
    Iran2002 = CubicSpline(new_list[101*select:101*(select+1), 2], new_list_labels.reshape(1, -1)[0][101*select:101*(select+1)])
    derivatives = []
    for i in range(13, 20):
        derivatives.append(derivative(Iran2002, i, 1/365))
    meanOfIraniansDerivatives.append(np.mean(np.array(derivatives)))

meanOfWorldsDerivatives = []
for select in range(72, 144):
    Iran2002 = CubicSpline(new_list[101*select:101*(select+1), 2], new_list_labels.reshape(1, -1)[0][101*select:101*(select+1)])
    derivatives = []
    for i in range(13, 20):
        derivatives.append(derivative(Iran2002, i, 1/365))
    meanOfWorldsDerivatives.append(np.mean(np.array(derivatives)))
print(np.arange(1950, 2022)[(np.array(meanOfIraniansDerivatives)-np.array(meanOfWorldsDerivatives)) > 0])

# جمعیت 2.5 ساله های جهان در سال 1950
WorldByAge = CubicSpline(new_list[101 * 0:101 * (0 + 1), 2],
                       new_list_labels.reshape(1, -1)[0][101 * 0:101 * (0 + 1)])
print(WorldByAge(2.5))

IranByAge = CubicSpline(new_list[101 * 72:101 * (72 + 1), 2],
                       new_list_labels.reshape(1, -1)[0][101 * 72:101 * (72 + 1)])

# نسبت جمعیت 87.5 ساله ها در ایران به کل جهان
print(IranByAge(87.5)/(WorldByAge(87.5))*100, "%")
