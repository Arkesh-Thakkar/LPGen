import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, Ridge
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
from z3 import *
import pulp
import itertools
from sympy import sympify, expand, Symbol, Mul
import re
import random
import warnings
import time
import math
warnings.filterwarnings("ignore")

def equality_invariants(X_original):

  final_coefs = []
  final_intercepts = []

  if X_original.shape[1] > 1:
    for i in range(X_original.shape[1]):
      X = np.delete(X_original[:], i, axis = 1)
      y = X_original[:, i]

      # Create a Lasso regressor
      regressor = Lasso(alpha = 0.1)
      
      # Train the regressor on the data
      regressor.fit(X, y)

      # Use SelectFromModel to select the non-zero coefficients
      model = SelectFromModel(regressor, prefit=True)
      
      # print(X)
      X_new = model.transform(X)

      if X_new.shape[1] == 0:
        continue
      
      masks = model.get_support()
      lin_reg= LinearRegression().fit(X_new, y)

      coefs = np.round(lin_reg.coef_,decimals = 2)
      intercepts = np.round(lin_reg.intercept_, decimals = 2)
      
      coef_ans = np.array([0.0 for _ in range(len(masks))])
      for mask_index in range(len(masks)):
        if str(masks[mask_index]) == "True":
          coef_ans[mask_index] = coefs[0]
          coefs = np.delete(coefs,0)
      coef_ans = np.insert(coef_ans,i , -1.0)
      
      large_value = 0
      for coef in coef_ans:
        if coef > 1000:
          large_value = 1
      
      if large_value == 0:
        final_coefs.append(coef_ans)
        final_intercepts.append(intercepts)

  if X_original.shape[1] > 1:
    for i in range(X_original.shape[1]):
      X_temp = np.delete(X_original, i, axis=1)
      y_temp = X_original[:,i]

      # create an instance of SelectKBest
      selector = SelectKBest(f_regression, k="all")
      selector.fit(X_temp, y_temp)

      scores = selector.scores_

      avg_score = np.average(scores)
      scores = filter(lambda score: score >= avg_score, list(scores))
    
      k = len(list(scores))
      
      # fit a linear regression model with the selected features
      selector = SelectKBest(f_regression, k=k)
      
      X_new = selector.fit_transform(X_temp, y_temp)
      masks = selector.get_support()

      lin_reg = LinearRegression()
      lin_reg.fit(X_new, y_temp)

      coefs = np.round(lin_reg.coef_,decimals = 2)
      intercepts = np.round(lin_reg.intercept_, decimals = 2)
      
      coef_ans = np.array([0.0 for _ in range(len(masks))])
      for mask_index in range(len(masks)):
        if str(masks[mask_index]) == "True":
          coef_ans[mask_index] = coefs[0]
          coefs = np.delete(coefs,0)
      coef_ans = np.insert(coef_ans,i , -1.0)
      
      large_value = 0
      for coef in coef_ans:
        if coef > 1000:
          large_value = 1
      
      if large_value == 0:
        final_coefs.append(coef_ans)
        final_intercepts.append(intercepts)


  final_coefs = np.array(final_coefs)
  final_intercepts = np.array(final_intercepts)

  equality_invariants_list = []
  for i in range(len(final_coefs)):
    # print(i)
    # print(not np.any(np.round(np.matmul(X_poly , final_coefs[i]) + final_intercepts[i], decimals = 2)))
    if not np.any(np.round(np.matmul(X_original , final_coefs[i]) + final_intercepts[i], decimals = 2)):
      equality_invariants_list.append(i)
  # print(equality_invariants_list)

  return equality_invariants_list, final_coefs, final_intercepts

def post_process_equality_simple_invariant(equality_invariant_list, final_coefs, final_intercepts, simply_equal_constraints, z3_vars, name):
  equality_invariants = []
  for i in range(len(equality_invariant_list)):
    inv = final_coefs[equality_invariant_list[i]][0] * z3_vars[0]
    for j in range(1,len(z3_vars)):
      inv += final_coefs[equality_invariant_list[i]][j] * z3_vars[j]
    equality_invariants.append(inv + final_intercepts[equality_invariant_list[i]] == 0)

  equality_invariants.extend(simply_equal_constraints)
  invariant = z3.simplify(z3.And(equality_invariants),arith_lhs=True, hoist_mul=True, elim_sign_ext=True, pull_cheap_ite=True)

  invariant_updated = []

  if 'And' in str(invariant):
    childs = invariant.children()
    for child in childs:
      if '*(' in str(child):
        sub_childs = child.children()[0].children()
        sub_child = '*'.join([str(sub_childs[i]) for i in range(len(sub_childs)-1)])
        sub_child += "*("+str(sub_childs[-1])+")"
        if sub_child.replace(" ","").replace("\n","") == str(child.children()[0]).replace(" ","").replace("\n",""):
          invariant_updated.append(sub_childs[-1] == child.children()[1])
        else:
          invariant_updated.append(child.children()[0] == child.children()[1])
      else:
        invariant_updated.append(child)
  else:
    if '*(' in str(invariant):
      sub_childs = invariant.children()[0].children()
      sub_child = '*'.join([str(sub_childs[i]) for i in range(len(sub_childs)-1)])
      sub_child += "*("+str(sub_childs[-1])+")"
      if sub_child.replace(" ","").replace("\n","") == str(invariant.children()[0]).replace(" ","").replace("\n",""):
        invariant_updated.append(sub_childs[-1] == invariant.children()[1])
      else:
        invariant_updated.append(invariant.children()[0] == invariant.children()[1])
    else:
      invariant_updated.append(invariant)

  # invariant = invariant_updated.copy()
  
  invariant = []
  pre, rec, post, lc, var_names_initial = pre_rec_post_conditions(name)
  
  vars_dict = {str(var): var for var in var_names_initial}
  
  invariant_mod = []
  for inv in invariant_updated:
    if str(inv) == 'True':
      continue
    if '%' not in str(inv):
      equations_strings = str(inv)
      equations_string = equations_strings.split('==')[0].strip()
      
      rhs = equations_strings.split('==')[1].strip()
      
      if re.match(r'^-?\d+(/\d+)?$', equations_string):
        equations_string, rhs = rhs, equations_string
      # Convert string to SymPy expression
      equations_expr = sympify(equations_string)

    
      numerator, denominator = equations_expr.as_numer_denom()
      
      numerator = expand(numerator)

      to_mul = int(denominator)

      if '/' in rhs:
        equation = str(numerator)+ '=='+str(int(rhs.split('/')[0])*int(to_mul))+'/'+str(rhs.split('/')[1])
      else:
        equation = str(numerator)+ '=='+str(int(rhs)*int(to_mul))

      
      equation = re.sub(r'ToReal\((.*?)\)',r'\1',equation)
      
      strings = equation.split('**')
      
      output_string = strings[0]
      for string in range(1,len(strings)):
        for mul in range(int(strings[string][0])-1):
          output_string += ('*'+output_string[-1])
        output_string += strings[string][1:]
      
      coefficients = re.findall(r"(\d+)", output_string)
      coefficients = [int(coeff) for coeff in coefficients]
      large_val = 0
      for coef in coefficients:
        if coef > 1000:
          large_val = 1
          break
      if large_val == 0:
        invariant.append(eval(output_string.split('==')[0], vars_dict) == eval(output_string.split('==')[1], vars_dict))
    else:
      invariant_mod.append(inv)
  
  if len(invariant) > 1:
    invariant_string = [str(inv) for inv in invariant]
    invariant_string = [re.sub(r'ToReal\((.*?)\)',r'\1',inv) for inv in invariant_string]
    invariant_string.sort(key = len, reverse = True)

    elements = [re.findall(r'\b(?:[A-Za-z]+\*)*[A-Za-z]+\b', inv) for inv in invariant_string]
    elements = [[ '*'.join(sorted(poly.split('*'))) for poly in element ] for element in elements]

    inv_dict = {str(elements[i]): invariant_string[i] for i in range(len(elements))}

    elements_copy = elements.copy()
    for element in elements_copy:
      if len(elements) > 1:
        elements.remove(element)
        to_check = elements[0].copy()
        for ele in elements[1:]:
          to_check.extend(ele)
        if len(element) == 1 or set(element) - set(to_check) != set():
          elements.append(element)

    invariant_string = [inv_dict[str(ele)] for ele in elements]
    # print(elements)
    # print(invariant_string)
    
    invariant = [eval(inv.split('==')[0], vars_dict) == eval(inv.split('==')[1], vars_dict) for inv in invariant_string]
  
    # print(invariant)
  if invariant_mod != []:
    invariant.extend(invariant_mod)

  if len(invariant) == 0:
    return [True]
  return invariant

import sys
def infer_2d_inequality(X):
  coef1, coef2, coef3, coef4 = ["error"], ["error"], ["error"], ["error"]
  inter1, inter2, inter3, inter4 = "error", "error", "error", "error"
  # Create a PuLP optimization problem
  model1 = pulp.LpProblem('Linear_Regression1', pulp.LpMinimize)
  model2 = pulp.LpProblem('Linear_Regression2', pulp.LpMinimize)
  model3 = pulp.LpProblem('Linear_Regression3', pulp.LpMinimize)
  model4 = pulp.LpProblem('Linear_Regression4', pulp.LpMinimize)

  # Define the decision variables
  w1 = pulp.LpVariable('w1', lowBound=1, upBound = 100000,cat = pulp.LpInteger)
  w2 = pulp.LpVariable('w2', lowBound=1, upBound = 100000,cat = pulp.LpInteger)

  w3 = pulp.LpVariable('w3', lowBound=1, upBound = 100000, cat = pulp.LpInteger)
  w4 = pulp.LpVariable('w4', lowBound =-100000,upBound = -1,cat = pulp.LpInteger)

  w5 = pulp.LpVariable('w5', lowBound =-100000,upBound = -1, cat = pulp.LpInteger)
  w6 = pulp.LpVariable('w6', lowBound=1, upBound = 100000,cat = pulp.LpInteger)

  w7 = pulp.LpVariable('w7', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w8 = pulp.LpVariable('w8', lowBound =-100000, upBound = -1,cat = pulp.LpInteger)

  x1 = pulp.LpVariable("x1", lowBound=0)
  x2 = pulp.LpVariable("x2", lowBound=0)
  x3 = pulp.LpVariable("x3", lowBound=0)
  x4 = pulp.LpVariable("x4", lowBound=0)
  
  b1 = pulp.LpVariable('b1', lowBound =-100000, upBound = 100000)
  b2 = pulp.LpVariable('b2', lowBound =-100000, upBound = 100000)
  b3 = pulp.LpVariable('b3', lowBound =-100000, upBound = 100000)
  b4 = pulp.LpVariable('b4', lowBound =-100000, upBound = 100000)
  
  # Define the objective function
  
  model1 += x1>= (1000000 + w1 + w2 + 0.2*b1)
  model1 += x1>= -(1000000 + w1 + w2 + 0.2*b1)

  model2 += x2>= (1000000 + 0.4 * (w3-w4) + 0.89 * w4 + 0.2 * b2)
  model2 += x2>= -(1000000 + 0.4 * (w3-w4) + 0.89 * w4 + 0.2 * b2)

  # model2 += x2>= (2*w3 + 3*w4 + b2)
  # model2 += x2>= -(2*w3 + 3*w4 + b2)
  
  model3 += x3>= (1000000 + 0.4 * (-w5+w6) +  0.89 * w5 + 0.2 * b3)
  model3 += x3>= -(1000000 + 0.4 * (-w5+w6) +  0.89 * w5 + 0.2 * b3)
  
  # model3 += x3>= (2*w6 + 3*w5 + b3)
  # model3 += x3>= -(2*w6 + 3*w5 + b3)

  model4 += x4>= (1000000 - w7 - w8 + 0.2*b4)
  model4 += x4>= -(1000000 - w7 - w8 + 0.2*b4)

  model1 += x1
  model2 += x2
  model3 += x3 
  model4 += x4

  # Add the constraints
  for i in range(min(500,len(X))):
      model1 += w1 * X[i][0] + w2 * X[i][1] + b1 >=  0
      model2 += w3 * X[i][0] + w4 * X[i][1] + b2 >=  0
      model3 += w5 * X[i][0] + w6 * X[i][1] + b3 >=  0
      model4 += w7 * X[i][0] + w8 * X[i][1] + b4 >=  0
  
  # Solve the optimization problem
  solver = pulp.GLPK_CMD(options=['--tmlim', '10'], msg = 0)
  # solver = pulp.GLPK_CMD(options=['--tmlim', '-1'])

  # Solve the optimization problem
  try:
      model1.solve(solver)
      if model1.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model1")
          pass
      else:
        coef1 = [int(w1.value()//math.gcd(w1.value(), w2.value(), int(b1.value()))), int(w2.value()//math.gcd(w1.value(),w2.value(), int(b1.value())))]
        inter1 = int(b1.value() //math.gcd(w1.value(), w2.value(), int(b1.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 1:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model1:", e)
      pass


  try:
      model2.solve(solver)
      if model2.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model2")
          pass
      else:
        coef2 = [int(w3.value()//math.gcd(w3.value(), -1*w4.value(), int(b2.value()))), int(w4.value()//math.gcd(w3.value(),-1*w4.value(), int(b2.value())))]
        inter2 = int(b2.value() // math.gcd(w3.value(), -1*w4.value(), int(b2.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 2:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 2:", e)
      pass

  
  try:
      model3.solve(solver)
      if model3.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model3")
          pass
      else:
        coef3 = [int(w5.value()//math.gcd(-1*w5.value(), w6.value(), int(b3.value()))), int(w6.value()//math.gcd(-1*w5.value(),w6.value(), int(b3.value())))]
        inter3 = int(b3.value()//math.gcd(-1*w5.value(), w6.value(), int(b3.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 3:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model3:", e)
      pass


  try:
      model4.solve(solver)
      if model4.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model4")
          pass
      else:
        coef4 = [int(w7.value()//math.gcd(-1*w7.value(), -1*w8.value(), int(b4.value()))), int(w8.value()//math.gcd(-1*w7.value(),-1*w8.value(), int(b4.value())))]
        inter4 = int(b4.value()//math.gcd(-1*w7.value(),-1*w8.value(), int(b4.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 4:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 4:", e)
      pass

  
  # Print the results
  return coef1, inter1, coef2, inter2, coef3, inter3, coef4, inter4

def infer_3d_inequality(X):
  coef1, coef2, coef3, coef4, coef5, coef6, coef7, coef8 = ["error"], ["error"], ["error"], ["error"], ["error"], ["error"], ["error"], ["error"]
  inter1, inter2, inter3, inter4, inter5, inter6, inter7, inter8 = "error", "error", "error", "error", "error", "error", "error", "error"
  # Create a PuLP optimization problem
  model1 = pulp.LpProblem('Linear_Regression1', pulp.LpMinimize)
  model2 = pulp.LpProblem('Linear_Regression2', pulp.LpMinimize)
  model3 = pulp.LpProblem('Linear_Regression3', pulp.LpMinimize)

  model4 = pulp.LpProblem('Linear_Regression4', pulp.LpMinimize)
  model5 = pulp.LpProblem('Linear_Regression5', pulp.LpMinimize)
  model6 = pulp.LpProblem('Linear_Regression6', pulp.LpMinimize)

  model7 = pulp.LpProblem('Linear_Regression7', pulp.LpMinimize)
  model8 = pulp.LpProblem('Linear_Regression8', pulp.LpMinimize)
  
  # Define the decision variables
  w1 = pulp.LpVariable('w1', lowBound= 1, upBound = 100000, cat = pulp.LpInteger)
  w2 = pulp.LpVariable('w2', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)
  w3 = pulp.LpVariable('w3', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)

  w4 = pulp.LpVariable('w4', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)
  w5 = pulp.LpVariable('w5', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w6 = pulp.LpVariable('w6', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)

  w7 = pulp.LpVariable('w7', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)
  w8 = pulp.LpVariable('w8', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)
  w9 = pulp.LpVariable('w9', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)

  w10 = pulp.LpVariable('w10', lowBound=1, upBound = 100000,cat = pulp.LpInteger)
  w11 = pulp.LpVariable('w11', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w12 = pulp.LpVariable('w12', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)

  w13 = pulp.LpVariable('w13',  lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w14 = pulp.LpVariable('w14', lowBound=1, upBound = 100000,cat = pulp.LpInteger)
  w15 = pulp.LpVariable('w15', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)

  w16 = pulp.LpVariable('w16', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w17 = pulp.LpVariable('w17', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w18 = pulp.LpVariable('w18', lowBound= 1, upBound = 100000,cat = pulp.LpInteger)

  w19 = pulp.LpVariable('w19', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w20 = pulp.LpVariable('w20', lowBound=1, upBound = 100000,cat = pulp.LpInteger)
  w21 = pulp.LpVariable('w21', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)

  w22 = pulp.LpVariable('w22', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w23 = pulp.LpVariable('w23', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)
  w24 = pulp.LpVariable('w24', lowBound =-100000, upBound = -1, cat = pulp.LpInteger)

  x1 = pulp.LpVariable("x1", lowBound=0)
  x2 = pulp.LpVariable("x2", lowBound=0)
  x3 = pulp.LpVariable("x3", lowBound=0)
  x4 = pulp.LpVariable("x4", lowBound=0)

  x5 = pulp.LpVariable("x5", lowBound=0)
  x6 = pulp.LpVariable("x6", lowBound=0)
  x7 = pulp.LpVariable("x7", lowBound=0)
  x8 = pulp.LpVariable("x8", lowBound=0)


  
  b1 = pulp.LpVariable('b1', lowBound =-100000, upBound = 100000)
  b2 = pulp.LpVariable('b2', lowBound =-100000, upBound = 100000)
  b3 = pulp.LpVariable('b3', lowBound =-100000, upBound = 100000)
  b4 = pulp.LpVariable('b4', lowBound =-100000, upBound = 100000)
  b5 = pulp.LpVariable('b5', lowBound =-100000, upBound = 100000)
  b6 = pulp.LpVariable('b6', lowBound =-100000, upBound = 100000)
  b7 = pulp.LpVariable('b7', lowBound =-100000, upBound = 100000)
  b8 = pulp.LpVariable('b8', lowBound =-100000, upBound = 100000)
  
  model1 += x1>= (1000000 + w1 + w2  + w3 + 0.2*b1)
  model1 += x1>= -(1000000 + w1 + w2  + w3 + 0.2*b1)

  model2 += x2>= (1000000 + 0.6*(w4-w5+w6) + 0.8*w5+0.2*b2)
  model2 += x2>= -(1000000 + 0.6*(w4-w5+w6) + 0.8*w5+0.2*b2)

  model3 += x3>= (1000000 + 0.6*(w7+w8-w9) + 0.8*w9+0.2*b3)
  model3 += x3>= -(1000000 + 0.6*(w7+w8-w9) + 0.8*w9+0.2*b3)
  
  model4 += x4>= (1000000 + 0.6*(w10-w11-w12) + 0.4*w11 + 0.4*w12+0.2*b4)
  model4 += x4>= -(1000000 + 0.6*(w10-w11-w12) + 0.4*w11 + 0.4*w12+0.2*b4)

  model5 += x5>= (1000000 + 0.6*(-w13+w14+w15) + 0.8*w13+0.2*b5)
  model5 += x5>= -(1000000 + 0.6*(-w13+w14+w15) + 0.8*w13+0.2*b5)
  
  model6 += x6>= (1000000 + 0.6*(-w16-w17+w18) + 0.4*w16+0.4*w17+0.2*b6)
  model6 += x6>= -(1000000 + 0.6*(-w16-w17+w18) + 0.4*w16+0.4*w17+0.2*b6)
  
  model7 += x7>= (1000000 + 0.6*(-w19+w20-w21) + 0.4*w19+0.4*w21+0.2*b7)
  model7 += x7>= -(1000000 + 0.6*(-w19+w20-w21) + 0.4*w19+0.4*w21+0.2*b7)
  
  model8 += x8>= (1000000 - w22 - w23 - w24 + 0.2*b8)
  model8 += x8>= -(1000000 - w22 - w23 - w24 + 0.2*b8)

  model1 += x1
  model2 += x2
  model3 += x3 
  model4 += x4

  model5 += x5
  model6 += x6
  model7 += x7 
  model8 += x8

  # Add the constraints
  for i in range(min(500,len(X))):
      model1 += w1 * X[i][0] + w2 * X[i][1]  + w3 * X[i][2] +  b1 >= 0
      model2 += w4 * X[i][0] + w5 * X[i][1]  + w6 * X[i][2] +  b2 >= 0
      model3 += w7 * X[i][0] + w8 * X[i][1]  + w9 * X[i][2] +  b3 >= 0
      model4 += w10 * X[i][0] + w11 * X[i][1]  + w12 * X[i][2] +  b4 >= 0
      model5 += w13 * X[i][0] + w14 * X[i][1]  + w15 * X[i][2] +  b5 >= 0
      model6 += w16 * X[i][0] + w17 * X[i][1]  + w18 * X[i][2] +  b6 >= 0
      model7 += w19 * X[i][0] + w20 * X[i][1]  + w21 * X[i][2] +  b7 >= 0
      model8 += w22 * X[i][0] + w23 * X[i][1]  + w24 * X[i][2] +  b8 >= 0

  # Solve the optimization problem
  solver = pulp.GLPK_CMD(options=['--tmlim', '10'], msg = 0)

  # Solve the optimization problem
  try:
      model1.solve(solver)
      if model1.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model1")
          pass
      else:
        coef1 = [int(w1.value()//math.gcd(w1.value(), w2.value(), w3.value(),int(b1.value()))), int(w2.value()//math.gcd(w1.value(),w2.value(), w3.value(),int(b1.value()))), int(w3.value()//math.gcd(w1.value(),w2.value(), w3.value(),int(b1.value())))]
        inter1 = int(b1.value() //math.gcd(w1.value(), w2.value(), w3.value(),int(b1.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 1:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model1:", e)
      pass


  try:
      model2.solve(solver)
      if model2.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model2")
          pass
      else:
        coef2 = [int(w4.value()//math.gcd(w4.value(), w5.value(), w6.value(),int(b2.value()))), int(w5.value()//math.gcd(w4.value(),w5.value(), w6.value(),int(b2.value()))), int(w6.value()//math.gcd(w4.value(),w5.value(), w6.value(),int(b2.value())))]
        inter2 = int(b2.value() //math.gcd(w4.value(), w5.value(), w6.value(),int(b2.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 2:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 2:", e)
      pass

  
  try:
      model3.solve(solver)
      if model3.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model3")
          pass
      else:
        coef3 = [int(w7.value()//math.gcd(w7.value(), w8.value(), w9.value(),int(b3.value()))), int(w8.value()//math.gcd(w7.value(),w8.value(), w9.value(),int(b3.value()))), int(w9.value()//math.gcd(w7.value(),w8.value(), w9.value(),int(b3.value())))]
        inter3 = int(b3.value() //math.gcd(w7.value(), w8.value(), w9.value(),int(b3.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 3:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 3:", e)
      pass
  
  try:
      model4.solve(solver)
      if model4.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model4")
          pass
      else:
        coef4 = [int(w10.value()//math.gcd(w10.value(), w11.value(), w12.value(),int(b4.value()))), int(w11.value()//math.gcd(w10.value(),w11.value(), w12.value(),int(b4.value()))), int(w12.value()//math.gcd(w10.value(),w11.value(), w12.value(),int(b4.value())))]
        inter4 = int(b4.value() //math.gcd(w10.value(), w11.value(), w12.value(),int(b4.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 4:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model4:", e)
      pass


  try:
      model5.solve(solver)
      if model5.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model5")
          pass
      else:
        coef5 = [int(w13.value()//math.gcd(w13.value(), w14.value(), w15.value(),int(b5.value()))), int(w14.value()//math.gcd(w13.value(),w14.value(), w15.value(),int(b5.value()))), int(w15.value()//math.gcd(w13.value(),w14.value(), w15.value(),int(b5.value())))]
        inter5 = int(b5.value() //math.gcd(w13.value(), w14.value(), w15.value(),int(b5.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 5:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 5:", e)
      pass

  
  try:
      model6.solve(solver)
      if model6.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model6")
          pass
      else:
        coef6 = [int(w16.value()//math.gcd(w16.value(), w17.value(), w18.value(),int(b6.value()))), int(w17.value()//math.gcd(w16.value(),w17.value(), w18.value(),int(b6.value()))), int(w18.value()//math.gcd(w16.value(),w17.value(), w18.value(),int(b6.value())))]
        inter6 = int(b6.value() //math.gcd(w16.value(), w17.value(), w18.value(),int(b6.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 6:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 6:", e)
      pass
  
  try:
      model7.solve(solver)
      if model7.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model7")
          pass
      else:
        coef7 = [int(w19.value()//math.gcd(w19.value(), w20.value(), w21.value(),int(b7.value()))), int(w20.value()//math.gcd(w19.value(),w20.value(), w21.value(),int(b7.value()))), int(w21.value()//math.gcd(w19.value(),w20.value(), w21.value(),int(b7.value())))]
        inter7 = int(b7.value() //math.gcd(w19.value(), w20.value(), w21.value(),int(b7.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 7:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 7:", e)
      pass

  try:
      model8.solve(solver)
      if model8.status != pulp.LpStatusOptimal:
          # print("Solver did not return an optimal solution for model8")
          pass
      else:
        coef8 = [int(w22.value()//math.gcd(w22.value(), w23.value(), w24.value(),int(b8.value()))), int(w23.value()//math.gcd(w22.value(),w23.value(), w24.value(),int(b8.value()))), int(w24.value()//math.gcd(w22.value(),w23.value(), w24.value(),int(b8.value())))]
        inter8 = int(b8.value() //math.gcd(w22.value(), w23.value(), w24.value(),int(b8.value())))
  except pulp.PulpError as e:
      # print("PuLP Error for model 8:", e)
      pass
  except ValueError as e:
      # print("Solver Error for model 8:", e)
      pass


  # Print the results
  return coef1, inter1, coef2, inter2, coef3, inter3, coef4, inter4, coef5, inter5, coef6, inter6, coef7, inter7, coef8, inter8

def generate_2d_inequality(X_original, columns):
  #vars_2d_original = list(new_data.columns[:])
  inequality_invariants_list = []
  vars_2d_original = list(columns.copy())
  
  vars_2d = [[vars_2d_original[ele1], vars_2d_original[ele2]] for ele1 in range(len(vars_2d_original)-1) for ele2 in range(ele1+1, len(vars_2d_original))]
  vars_2d = [list(pair) for pair in vars_2d if pair[0] != pair[1]]

  constrs = []
  for var_pair in vars_2d:
    
    index1, index2 = vars_2d_original.index(var_pair[0]), vars_2d_original.index(var_pair[1])
    coef1, inter1, coef2, inter2, coef3, inter3, coef4, inter4 = infer_2d_inequality(X_original[:,[index1, index2]])

    if inter1 != "error" :
      if np.all(X_original[:,index1] * coef1[0] + X_original[:, index2] * coef1[1] + inter1 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef1[0] + Int(var_pair[1]) * coef1[1] + inter1 >= 0)
    if inter2 != "error" :
      if np.all(X_original[:,index1] * coef2[0] + X_original[:, index2] * coef2[1] + inter2 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef2[0] + Int(var_pair[1]) * coef2[1] + inter2 >= 0)

    if inter3 != "error" :
      if np.all(X_original[:,index1] * coef3[0] + X_original[:, index2] * coef3[1] + inter3 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef3[0] + Int(var_pair[1]) * coef3[1] + inter3 >= 0)
    if inter4 != "error" :
      if np.all(X_original[:,index1] * coef4[0] + X_original[:, index2] * coef4[1] + inter4 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef4[0] + Int(var_pair[1]) * coef4[1] + inter4 >= 0)
    
  return inequality_invariants_list

def generate_3d_inequality(X_original, columns):
  # vars_3d_original = list(new_data.columns[:])
  vars_3d_original = list(columns.copy())
  vars_3d = [[vars_3d_original[ele1], vars_3d_original[ele2], vars_3d_original[ele3]] for ele1 in range(len(vars_3d_original)-2) for ele2 in range(ele1+1, len(vars_3d_original)-1) for ele3 in range(ele2+1, len(vars_3d_original))]

  inequality_invariants_list = []
  constrs = []
  for var_pair in vars_3d:
    index1, index2, index3 = vars_3d_original.index(var_pair[0]), vars_3d_original.index(var_pair[1]), vars_3d_original.index(var_pair[2])
    coef1, inter1, coef2, inter2, coef3, inter3, coef4, inter4, coef5, inter5, coef6, inter6, coef7, inter7, coef8, inter8 = infer_3d_inequality(X_original[:,[index1, index2, index3]])
    

    if inter1 != "error" :
      if np.all(X_original[:,index1] * coef1[0] + X_original[:, index2] * coef1[1] + X_original[:, index3] * coef1[2] + inter1 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef1[0] + Int(var_pair[1]) * coef1[1] + Int(var_pair[2]) * coef1[2] + inter1 >= 0)
    
    if inter2 != "error" :
      if np.all(X_original[:,index1] * coef2[0] + X_original[:, index2] * coef2[1] + X_original[:, index3] * coef2[2] + inter2 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef2[0] + Int(var_pair[1]) * coef2[1] + Int(var_pair[2]) * coef2[2] + inter2 >= 0)
    
    if inter3 != "error" :
      if np.all(X_original[:,index1] * coef3[0] + X_original[:, index2] * coef3[1] + X_original[:, index3] * coef3[2] + inter3 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef3[0] + Int(var_pair[1]) * coef3[1] + Int(var_pair[2]) * coef3[2] + inter3 >= 0)

    if inter4 != "error" :
      if np.all(X_original[:,index1] * coef4[0] + X_original[:, index2] * coef4[1] + X_original[:, index3] * coef4[2] + inter4 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef4[0] + Int(var_pair[1]) * coef4[1] + Int(var_pair[2]) * coef4[2] + inter4 >= 0)
    
    if inter5 != "error" :
      if np.all(X_original[:,index1] * coef5[0] + X_original[:, index2] * coef5[1] + X_original[:, index3] * coef5[2] + inter5 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef5[0] + Int(var_pair[1]) * coef5[1] + Int(var_pair[2]) * coef5[2] + inter5 >= 0)
    
    if inter6 != "error" :
      if np.all(X_original[:,index1] * coef6[0] + X_original[:, index2] * coef6[1] + X_original[:, index3] * coef6[2] + inter6 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef6[0] + Int(var_pair[1]) * coef6[1] + Int(var_pair[2]) * coef6[2] + inter6 >= 0)
    
    if inter7 != "error" :
      if np.all(X_original[:,index1] * coef7[0] + X_original[:, index2] * coef7[1] + X_original[:, index3] * coef7[2] + inter7 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef7[0] + Int(var_pair[1]) * coef7[1] + Int(var_pair[2]) * coef7[2] + inter7 >= 0)

    if inter8 != "error" :
      if np.all(X_original[:,index1] * coef8[0] + X_original[:, index2] * coef8[1] + X_original[:, index3] * coef8[2] + inter8 >= 0):
        inequality_invariants_list.append(Int(var_pair[0]) * coef8[0] + Int(var_pair[1]) * coef8[1] + Int(var_pair[2]) * coef8[2] + inter8 >= 0)
  return inequality_invariants_list

def generate_complete_invariant(invariant, inequality_2d, inequality_3d, min_max_invariants):
  invariant_list = []
  invariant_list.extend(invariant)
  invariant_list.extend(inequality_2d)
  invariant_list.extend(inequality_3d)
  invariant_list.extend(min_max_invariants)

  return invariant_list

def pre_rec_post_conditions(name):
  global external_file
  global var_names_initial
  if external_file == 0:
    ######################################### Zilu ####################################################
    if name == "benchmark01_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x == 1, y == 1)
      rec= And(x2 == x+y, y2 == x2)
      post = And(y>=1)
    if name == "benchmark02_linear":
      n, i, l = Ints('n i l')
      var_names_initial = [n, i, l]
      n2, i2, l2 = Ints('n2 i2 l2')
      lc = i<n
      pre = And(i == l, l > 0)
      rec = And(i2 == i+1, l2 == l, n2 == n)
      post = And(l>=1)
    if name == "benchmark03_linear":
      x, y, i, j = Ints('x y i j')
      var_names_initial = [x, y, i, j]
      x2, y2, i2, j2 = Ints('x2 y2 i2 j2')
      lc = unknown
      pre = And(x == 0, y == 0, i == 0, j == 0)
      rec = And(x2 == x+1, y2 == y+1, i2 == i+x2, Or(j2 == j+y2, j2 == j+y2+1))
      post = And(j>=i)
    if name == "benchmark04_conjunctive":
      k, j, n = Ints('k j n')
      var_names_initial = [k, j, n]
      k2, j2, n2 = Ints('k2 j2 n2')
      lc = j <= n-1
      pre = And(n >= 1, k >= n, j == 0)
      rec = And(j2 == j+1, k2 == k-1, n2 == n)
      post = And(k>=0)
    if name == "benchmark05_conjunctive":
      x, y, n = Ints('x y n')
      var_names_initial = [x, y, n]
      x2, y2, n2 = Ints('x2 y2 n2')
      lc = x < n
      pre = And(x >= 0, x <= y, y < n)
      rec = And(x2 == x+1, If(x2 > y, y2 == y+1, y2 == y), n2 == n)
      post = And(y == n)
    if name == "benchmark06_conjunctive":
      i, j, x, y, k = Ints('i j x y k')
      var_names_initial = [i, j, x, y, k]
      i2, j2, x2, y2, k2 = Ints('i2 j2 x2 y2 k2')
      lc = unknown
      pre = And(j == 0, x + y == k)
      rec = And(If(j == i, And(x2 == x+1, y2 == y-1), And(x2 == x-1, y2 == y+1)), j2 == j+1, i2 == i, k2 == k)
      post = And(x+y == k)
    if name == "benchmark07_linear":
      i, n, k = Ints('i n k')
      var_names_initial = [i, n, k]
      i2, n2, k2 = Ints('i2 n2 k2')
      lc = i < n
      pre = And(i == 0, n > 0, n < 10, k >= -1998)
      rec = And(i2 == i+1, Or(k2 == k+2000, k2 == k+4000), n2 == n)
      post = And(k > n)
    if name == "benchmark08_conjunctive":
      n, sum, i = Ints('n sum i')
      var_names_initial = [n, sum, i]
      n2, sum2, i2 = Ints('n2 sum2 i2')
      lc = i < n
      pre = And(i == 0, sum == 0, n >= 0)
      rec = And(sum2 == sum + i, i2 == i+1, n2 == n)
      post = And(sum >= 0)
    if name == "benchmark09_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x!=0
      pre = And(x == y, y>=0)
      rec = And(x2 == x-1, y2 == y-1)
      post = And(y == 0)
    if name == "benchmark10_conjunctive":
      i, c = Ints('i c')
      var_names_initial = [i, c]
      i2, c2 = Ints('i2 c2')
      lc = i < 100
      pre = And(i == 0, c == 0)
      rec = And(c2 == c+i, i2 == i+1)
      post = And(c >= 0)
    if name == "benchmark11_linear":
      x, n = Ints('x n')
      var_names_initial = [x, n]
      x2, n2 = Ints('x2 n2')
      lc = x < n
      pre = And(x == 0, n>0)
      rec = And(x2 == x+1, n2 == n)
      post = And(x == n)
    if name == "benchmark12_linear":
      x, y, t = Ints('x y t')
      var_names_initial = [x, y, t]
      x2, y2, t2 = Ints('x2 y2 t2')
      lc = unknown
      pre = And(x != t, y == t)
      rec = And(If(x > 0, y2 == y+x, y2 == y), x2 == x, t2 == t)
      post = And(y >= t)
    if name == "benchmark13_conjunctive":
      i, j, k = Ints('i j k')
      var_names_initial = [i, j, k]
      i2, j2, k2 = Ints('i2 j2 k2')
      lc = i<=k
      pre = And(i == 0, j == 0)
      rec = And(i2 == i + 1, j2 == j+1, k2 == k)
      post = And(i == j)
    if name == "benchmark14_linear":
      i = Int('i')
      var_names_initial = [i]
      i2 = Int('i2')
      lc = i>0
      pre = And(i>=0, i<=200)
      rec = And(i2 == i-1)
      post = And(i >= 0)
    if name == "benchmark15_conjunctive":
      low, mid, high = Ints('low mid high')
      var_names_initial = [low, mid, high]
      low2, mid2, high2 = Ints('low2 mid2 high2')
      lc = mid > 0
      pre = And(low == 0, mid >= 1, high == 2*mid)
      rec = And(low2 == low+1, high2 == high-1, mid2 == mid - 1)
      post = And(low == high)
    if name == "benchmark16_conjunctive":
      i, k = Ints('i k')
      var_names_initial = [i, k]
      i2, k2 = Ints('i2 k2')
      lc = unknown
      pre = And(k>=0, k<=1, i==1)
      rec = And(i2 == i+1, k2 == k-1)
      post = And(i + k >=1, i+k<=2, i>=1)
    if name == "benchmark17_conjunctive":
      i, k, n = Ints('i k n')
      var_names_initial = [i, k, n]
      i2, k2, n2 = Ints('i2 k2 n2')
      lc = i<n
      pre = And(i==0, k==0)
      rec = And(i2 == i+1, k2 == k+1, n2 == n)
      post = And(k>=n)
    if name == "benchmark18_conjunctive":
      i, k, n = Ints('i k n')
      var_names_initial = [i, k, n]
      i2, k2, n2 = Ints('i2 k2 n2')
      lc = i<n
      pre = And(i == 0, k == 0, n > 0)
      rec = And(i2 == i+1, k2 == k+1, n2 == n)
      post = And(i == k, k == n)
    if name == "benchmark19_conjunctive":
      j, k, n = Ints('j k n')
      var_names_initial = [j, k, n]
      j2, k2, n2 = Ints('j2 k2 n2')
      lc = And(j>0, n>0)
      pre = And(j==n, k==n, n>0)
      rec = And(j2 == j-1, k2 == k-1, n2 == n)
      post = And(k==0)
    if name == "benchmark20_conjunctive":
      i, n, sum = Ints('i n sum')
      var_names_initial = [i, n, sum]
      i2, n2, sum2 = Ints('i2 n2 sum2')
      lc = i<n
      pre = And(i == 0, n>=0, n<=100, sum == 0)
      rec = And(sum2 == sum + i, i2 == i+1, n2 == n)
      post = And(sum >= 0)
    if name == "benchmark21_disjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x+y<=-2
      pre = Or(x>0, y>0)
      rec = And(If(x>0, And(x2 == x+1, y2 == y), And(x2 == x, y2 == y+1)))
      post = Or(x>0, y>0)
    if name == "benchmark22_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x == 1, y==0)
      rec = And(x2 == x+y, y2 == y+1)
      post = And(x >= y)
    if name == "benchmark23_conjunctive":
      i, j = Ints('i j')
      var_names_initial = [i, j]
      i2, j2 = Ints('i2 j2')
      lc = i<100
      pre = And(i == 0, j == 0)
      rec = And(j2 == j+2, i2 == i+1)
      post = And(j == 200)
    if name == "benchmark24_conjunctive":
      i, k, n = Ints('i k n')
      var_names_initial = [i, k, n]
      i2, k2, n2 = Ints('i2 k2 n2')
      lc = i<n
      pre = And(i==0, k==n, n>=0)
      rec = And(k2 == k-1, i2 == i+2, n2 == n)
      post = And(2*k >= n-1)
    if name == "benchmark25_linear":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = x < 10
      pre = And(x<0)
      rec = And(x2 == x+1)
      post = And(x == 10)
    if name == "benchmark26_linear":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x<y
      pre = And(x<y)
      rec = And(x2 == x+1, y2 == y)
      post = And(x == y)
    if name == "benchmark27_linear":
      i, j, k = Ints('i j k')
      var_names_initial = [i, j, k]
      i2, j2, k2 = Ints('i2 j2 k2')
      lc = i<j
      pre = And(i<j, k>i-j)
      rec = And(k2 == k+1, i2 == i+1, j2 == j)
      post = And(k>0)
    if name == "benchmark28_linear":
      i, j = Ints('i j')
      var_names_initial = [i, j]
      i2, j2 = Ints('i2 j2')
      lc = i < j
      pre = And(i*i < j*j, i>=1, j>=2)
      rec = And(If(j<2*i, And(i2 == j-i, j2 == j-i2), And(i2 == i, j2 == j-i)))
      post = And(j == i)
    if name == "benchmark29_linear":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x < y
      pre = And(x < y)
      rec = And(x2 == x+100, y2 == y)
      post = And(x>=y, x<=y+99)
    if name == "benchmark30_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x == y)
      rec = And(x2 == x+1, y2 == y+1)
      post = And(x == y)
    if name == "benchmark31_disjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x<0
      pre = And(x < 0)
      rec = And(x2 == x+y, y2 == y+1)
      post = And(y>=0)
    if name == "benchmark32_linear":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = unknown
      pre = Or(x==1, x==2)
      rec = And(If(x==1, x2 == 2, If(x == 2, x2 == 1, True), True))
      post = And(x <= 8)
    if name == "benchmark33_linear":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = And(x>=0, x<100)
      pre = And(x>=0)
      rec = And(x2 == x+1)
      post = And(x>=100)
    if name == "benchmark34_conjunctive":
      j, k, n = Ints('j k n')
      var_names_initial = [j, k, n]
      j2, k2, n2 = Ints('j2 k2 n2')
      lc = And(j<n, n>0)
      pre = And(j == 0, k == n, n>0)
      rec = And(j2 == j+1, k2 == k-1, n2 == n)
      post = And(k == 0)
    if name == "benchmark35_linear":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = And(x>=0, x<10)
      pre = And(x>=0)
      rec = And(x2 == x+1)
      post = And(x >= 10)
    if name == "benchmark36_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x == y, y == 0)
      rec = And(x2 == x+1, y2 == y+1)
      post= And(x == y, y>=0)
    if name == "benchmark37_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x > 0
      pre = And(x == y, x>=0)
      rec = And(x2 == x-1, y2 == y-1)
      post = And(y>=0)
    if name == "benchmark38_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x == y, y == 0)
      rec = And(x2 == x+4, y2 == y+1)
      post = And(x == 4*y, x>=0)
    if name == "benchmark39_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x>0
      pre = And(x == 4*y, x>=0)
      rec = And(x2 == x-4, y2 == y-1)
      post = And(y>=0)
    if name == "benchmark40_polynomial":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x*y>=0)
      rec = And(If(x == 0, If(y > 0, And(x2 == x+1, y2 == y), And(x2 == x-1, y2 == y)), If(x>0, And(y2 == y+1, x2 == x), And(x2 == x-1, y2 == y))))
      post = And(x*y>=0)
    if name == "benchmark41_conjunctive":
      x, y, z = Ints('x y z')
      var_names_initial = [x, y, z]
      x2, y2, z2 = Ints('x2 y2 z2')
      lc = unknown
      pre = And(x == y, y == 0, z == 0)
      rec = And(x2 == x+1, y2 == y+1, z2 == z-2)
      post = And(x == y, x>=0, x+y+z == 0)
    if name == "benchmark42_conjunctive":
      x, y, z = Ints('x y z')
      var_names_initial = [x, y, z]
      x2, y2, z2 = Ints('x2 y2 z2')
      lc = x > 0
      pre = And(x == y, x>=0, x+y+z == 0)
      rec = And(x2 == x-1, y2 == y-1, z2 == z+2)
      post = And(z<=0)
    if name == "benchmark43_conjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = And(x<100, y<100)
      pre = And(x<100, y<100)
      rec = And(x2 == x+1, y2 == y+1)
      post = Or(x == 100, y==100)
    if name == "benchmark44_disjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x<y
      pre = And(x < y)
      rec = And(If(And(x<0, y<0), And(x2 == x+7, y2 == y-10), If(And(x<0, y>=0), And(x2 == x+7, y2 == y+3), And(x2 == x+10, y2 == y+3))))
      post = And(x>=y, x<=y+16)
    if name == "benchmark45_disjunctive":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = Or(y>0, x>0)
      rec = And(If(x>0, And(x2 == x+1, y2 == y), And(x2 == x, y2 == y+1)))
      post = Or(x>0, y>0)
    if name == "benchmark46_disjunctive":
      x, y, z = Ints('x y z')
      var_names_initial = [x, y, z]
      x2, y2, z2 = Ints('x2 y2 z2')
      lc = unknown
      pre = Or(x>0, y>0, z>0)
      rec = And(If(x>0, x2 == x+1, x2 == x), If(y>0, And(y2 == y+1, z2 == z), And(y2 == y, z2 == z+1)))
      post = Or(x>0, y>0, z>0)
    if name == "benchmark47_linear":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = x<y
      pre = And(x<y)
      rec = And(If(x<0, x2 == x+7, x2 == x+10), If(y<0, y2 == y-10, y2 == y+3))
      post = And(x>=y, x<=y+16)
    if name == "benchmark48_linear":
      i, j, k = Ints('i j k')
      var_names_initial = [i, j, k]
      i2, j2, k2 = Ints('i2 j2 k2')
      lc = i<j
      pre = And(i<j, k>0)
      rec =And(k2 == k+1, i2 == i+1, j2 == j)
      post = And(k>j-i)
    if name == "benchmark49_linear":
      i, j, r = Ints('i j r')
      var_names_initial = [i, j, r]
      i2, j2, r2 = Ints('i2 j2 r2')
      lc = i > 0
      pre = And(r > i+j)
      rec = And(i2 == i-1, j2 == j+1, r2 == r)
      post = And(r > i+j)
    if name == "benchmark50_linear":
      xa, ya = Ints('xa ya')
      var_names_initial = [xa, ya]
      xa2, ya2 = Ints('xa2 ya2')
      lc = xa > 0
      pre = And(xa + ya > 0)
      rec = And(xa2 == xa-1, ya2 == ya+1)
      post = And(ya >= 0)
    if name == "benchmark51_polynomial":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = unknown
      pre = And(x>=0, x<=50)
      rec = And(If(Or(x>50,x==0), x2 == x+1, x2 == x-1))
      post = And(x>=0, x<=50)
    if name == "benchmark52_polynomial":
      i = Int('i')
      var_names_initial = [i]
      i2 = Int('i2')
      lc = i * i < 100
      pre = And(i > -10, i < 10)
      rec = And(i2 == i+1)
      post = And(i==10)
    if name == "benchmark53_polynomial":
      x, y = Ints('x y')
      var_names_initial = [x, y]
      x2, y2 = Ints('x2 y2')
      lc = unknown
      pre = And(x * y >= 0)
      rec = And(If(x==0, If(y>0, And(x2 == x+1, y2 == y+1), And(x2 == x-2, y2 == y)), If(x>0, And(y2 == y+1, x2 == x), And(y2 == y, x2 == x-1))))
      post = And(x*y>=0)
    

    ############################# SVCOMP Loop Inv ####################################
    if name == "bin-suffix-5":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = x <= 2000
      pre = And(x == 5)
      rec = And(x2 == x+8)
      post = And((Int2BV(x,16) & 5) == 5)
    if name == "const":
      s = Int('s')
      var_names_initial = [s]
      s2 = Int('s2')
      lc = unknown
      pre = And(s==0)
      rec = And(If(s!=0, s2 == s+1, s2 == s))
      post = And(s == 0)
    if name == "eq1":
      w,x,y,z = Ints('w x y z')
      var_names_initial = [w,x,y,z]
      w2,x2,y2,z2 = Ints('w2 x2 y2 z2')
      lc = unknown
      pre = And(x == w, z == y)
      rec = And(Or(And(x2 == x+1, w2 == w+1, y2 == y, z2 == z), And(x2 == x, w2 == w, y2 == y+1, z2 == z+1)))
      post = And(w == x, y==z)
    if name == "eq2":
      w,x,y,z = Ints('w x y z')
      var_names_initial = [w,x,y,z]
      w2,x2,y2,z2 = Ints('w2 x2 y2 z2')
      lc = unknown
      pre = And(x == w, y == w+1, z == x+1)
      rec = And(y2 == y+1, z2 == z+1, w2 == w, x2 == x)
      post = And(y == z)
    if name == "even":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = unknown
      pre = And(x == 0)
      rec = And(x2 == x+2)
      post = And(x%2 == 0)
    if name == "linear-inequality-inv-a":
      n, v, s, i = Ints('n v s i')
      var_names_initial = [n, v, s, i]
      n2, v2, s2, i2 = Ints('n2 v2 s2 i2')
      lc = i < n
      pre = And(v== 0, s == 0, i == 0, n >=0, n<=255)
      rec = And(v2 >=0, v2<=255, s2 == s+v2, i2== i+1, n2 == n, v2*i2 >= s2)
      post = And(s>=v, s<=65025)
    if name == "linear-inequality-inv-b":
      n, v, s, i = Ints('n v s i')
      var_names_initial = [n, v, s, i]
      n2, v2, s2, i2 = Ints('n2 v2 s2 i2')
      lc = i < n
      pre = And(v== 0, s == 0, i == 0, n >=0, n<=255)
      rec = And(v2 >=0, v2<=255, s2 == (s+v2)%256, i2== i+1, n2 == n)
      post = And(s>=0, s<=255)
    if name == "mod4":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = unknown
      pre = And(x == 0)
      rec = And(x2 == x+4)
      post = And(x%4 == 0)
    if name == "odd":
      x = Int('x')
      var_names_initial = [x]
      x2 = Int('x2')
      lc = unknown
      pre = And(x==1)
      rec = And(x2 == x+2)
      post = And(x%2 == 1)
    
    ################################################### NLA #####################################################
    
    if name == "cohencu_data":
      var_names_initial = [Int('a'), Int('n'), Int('x'), Int('y'), Int('z')]
      a, n, x, y, z = Int('a'), Int('n'), Int('x'), Int('y'), Int('z')
      a2, n2, x2, y2, z2 = Int('a2'), Int('n2'), Int('x2'), Int('y2'), Int('z2')

      pre = And(n == 0, x == 0, y == 1, z == 6, a >= 0)
      rec = And(n2 == n + 1, x2 == x + y, y2 == y + z, z2 == z + 6, a2 == a)
      post = And(x == (a + 1) * (a + 1) * (a + 1))
      lc = n <= a

    if name == "cohendiv_1":
      var_names_initial = [Int('x'), Int('y'), Int('q'), Int('a'), Int('b'), Int('r')]
      x, y, q, a, b, r = Int('x'), Int('y'), Int('q'), Int('a'), Int('b'), Int('r')
      x2, y2, q2, a2, b2, r2 = Int('x2'), Int('y2'), Int('q2'), Int('a2'), Int('b2'), Int('r2')

      lc = r >= y
      pre = And(x > 0, y > 0, q == 0, r == x, a == 0, b == 0)
      rec = And(r2 == r-b2, q2 == q + a2, x2 == x, y2 ==y, If(r >= 2*y, And(a2 * y == 2 * r, b2 == y*a2 ), And(a2 == 1, b2 == y)))
      post = And(x == q * y + r,  r < y)

    if name == "cohendiv_2":
      var_names_initial = [Int('x'), Int('y'), Int('q'), Int('a'), Int('b'), Int('r')]
      x, y, q, a, b, r = Int('x'), Int('y'), Int('q'), Int('a'), Int('b'), Int('r')
      x2, y2, q2, a2, b2, r2 = Int('x2'), Int('y2'), Int('q2'), Int('a2'), Int('b2'), Int('r2')

      lc = r >= 2 * b
      pre = And(a==1, b == y, r >= y, q==0, r == x, x > 0, y>0)
      rec = And(x2 == x, y2 == y, q2 == q, r2 == r, a2 == 2 * a, b2 == 2 * b)
      post = And(b == y * a, x == q * y + r, r >= 0, x >= 1, y >= 1, r < 2 * b)

    if name == "dijkstra_1":
      var_names_initial = [Int('r'), Int('p'), Int('n'), Int('q'), Int('h')]
      r, p, n, q, h = Int('r'), Int('p'), Int('n'), Int('q'), Int('h')
      r2, p2, n2, q2, h2 = Int('r2'), Int('p2'), Int('n2'), Int('q2'), Int('h2')

      lc = q <= n
      pre = And(p == 0, q == 1, r == n, h == 0, n >= 0)
      rec = And(p2 == p, q2 == 4 * q, r2 == r, n2 == n, h2 == h)
      post = And(p == 0, q > n, r == n, h == 0, n >= 0)

    if name == "dijkstra_2":
      var_names_initial = [Int('r'), Int('p'), Int('n'), Int('q'), Int('h')]
      r, p, n, q, h = Int('r'), Int('p'), Int('n'), Int('q'), Int('h')
      r2, p2, n2, q2, h2 = Int('r2'), Int('p2'), Int('n2'), Int('q2'), Int('h2')

      lc = q != 1
      pre = And(p == 0, q > n, r == n, h == 0, n >= 0)
      rec = And(q == 4 * q2, h2 == p + q2, n2 == n,
                Or(And(r >= h2,  p == 2 * p2 - 2 * q2, r2 == r - h2),
                    And(r < h2,  p == 2 * p2, r2 == r)))
      post = And(p * p <= n, (p + 1) * (p + 1) > n)

    if name == "divbin_1":
      var_names_initial = [Int('A'), Int('B'), Int('q'), Int('r'), Int('b')]
      A, B, q, r, b = Int('A'), Int('B'), Int('q'), Int('r'), Int('b')
      A2, B2, q2, r2, b2 = Int('A2'), Int('B2'), Int('q2'), Int('r2'), Int('b2')

      lc = r >= b
      pre = And(A > 0, B > 0, q == 0, r == A, b == B)
      rec = And(A2 == A, B2 == B, q2 == q, r2 == r, b2 == 2 * b)
      post = And(q == 0, A == r, b > 0, r > 0, r < b)

    if name == "divbin_2":
      var_names_initial = [Int('A'), Int('B'), Int('q'), Int('r'), Int('b')]
      A, B, q, r, b = Int('A'), Int('B'), Int('q'), Int('r'), Int('b')
      A2, B2, q2, r2, b2 = Int('A2'), Int('B2'), Int('q2'), Int('r2'), Int('b2')

      lc = b != B
      pre = And(q == 0, A == r, b > 0, r > 0, r < b)
      rec = And(A2 == A, B2 == B, b == 2 * b2,
                Or(And(r >= b2, q2 == 2 * q + 1, r2 == r - b2),
                    And(r < b2, q2 == 2 * q, r2 == r)))
      temp = Int('temp')
      post = Exists(temp, And(temp >= 0, temp < B, A == q * B + temp))

    if name == "egcd_1_data":
      var_names_initial = [Int('x'), Int('y'), Int('a'), Int('b'), Int('p'), Int('r'), Int('q'), Int('s')]
      x, y, a, b, p, r, q, s = Int('x'), Int('y'), Int('a'), Int('b'), Int('p'), Int('r'), Int('q'), Int('s')
      x2, y2, a2, b2, p2, r2, q2, s2 = Int('x2'), Int('y2'), Int('a2'), Int('b2'), Int('p2'), Int('r2'), Int('q2'), Int('s2')

      lc = a != b
      pre = And(x >= 1, y >= 1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)
      rec = And(x2 == x, y2 == y,If(a>b, And(a2 == a - b, p2 == p - q, r2 == r - s, b2 == b, q2 == q, r2 == r), And(b2 == b - a, q2 == q - p, s2 == s - r, a2 == a, p2 == p, r2 == r)))
      post = And(p * s - r * q == 1, y * r + x * p == a, x * q + y * s == b, a == b)

    if name == "egcd2_1":
      a, b, p, q, r, s, x, y = Ints('a b p q r s x y')
      var_names_initial = [a, b, p, q, r, s, x, y]
      a2, b2, p2, q2, r2, s2, x2, y2 = Ints('a2 b2 p2 q2 r2 s2 x2 y2')

      lc = b!=0
      pre = And(x>=1, y>=1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)
      c = Int('c')
      rec = And(If(a>=b, c == a - b * (a / b), c == a), a2 == b, b2 == c, p2 == q, q2 == p - q * (a / b), r2 == s, s2 == r - s * (a / b), x2 == x, y2 == y)
      post = And(True)

    if name == "egcd2_2_data":
      a, b, c, k, p, q, r, s, x, y = Ints('a b c k p q r s x y')
      var_names_initial = [a, b, c, k, p, q, r, s, x, y]
      a2, b2, c2, k2, p2, q2, r2, s2, x2, y2 = Ints('a2 b2 c2 k2 p2 q2 r2 s2 x2 y2')

      lc = c>=b
      pre = And(x>=1, y>=1, a == x, b == y, p == 1, q == 0, r == 0, s == 1, k == 0, c == a)
      rec = And(c2 == c - b, k2 == k+1, a2 == a, b2 == b, p2 == p, q2 == q, r2 == r, s2 == s, x2 == x, y2 == y)
      post = And(a == k*b + c, a == y*r+x*p, b == x*q+y*s)


    if name == "egcd3_1_data":
      a, b, p, q, r, s, x, y = Ints('a b p q r s x y')
      var_names_initial = [a, b, p, q, r, s, x, y]
      a2, b2, p2, q2, r2, s2, x2, y2 = Ints('a2 b2 p2 q2 r2 s2 x2 y2')

      lc = b!=0
      pre = And(x>=1, y>=1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)
      rec = And(a2 == b, b2 == a, p2 == q, q2 == p, r2 == s, s2 == r, a < b, x2 == x, y2 == y)
      post = And(a == y*r + x*p, b == x*q + y*s)


    if name == "egcd3_2_data":
      a, b, p, q, r, s, x, y, c, k = Ints('a b p q r s x y c k')
      var_names_initial = [a, b, p, q, r, s, x, y, c, k]
      a2, b2, p2, q2, r2, s2, x2, y2, c2, k2 = Ints('a2 b2 p2 q2 r2 s2 x2 y2 c2 k2')

      lc = c >= b
      pre = And(c == a, k==0, x>=1, y>=1, a == y*r + x*p, b == x*q + y*s)
      rec = And(c2 == c - b, k2 == k + 1, x2 == x, y2 == y, s2 == s, r2 == r, q2 == q, p2 == p, a2 == a, b2 == b, c < 2*b)
      post = And(a == y*r + x*p, b == x*q + y*s, a == k*b + c)

    if name == "egcd3_3_data":
      a, b, p, q, r, s, x, y, c, k, d, v = Ints('a b p q r s x y c k d v')
      var_names_initial = [a, b, p, q, r, s, x, y, c, k, d, v]
      a2, b2, p2, q2, r2, s2, x2, y2, c2, k2, d2, v2 = Ints('a2 b2 p2 q2 r2 s2 x2 y2 c2 k2 d2 v2')

      lc = c >= 2*v
      pre = And(d == 1, v == b, x>=1, y>=1, b == x*q + y*s, a == y*r + x*p, a == k*b + c)
      rec = And(d2 == 2*d, v2 == 2*v, a2 == a, b2 == b, p2 == p, q2 == q, r2 == r, s2 == s, x2 == x, y2 == y, c2 == c, k2 == k)
      post = And(a == y*r + x*p, b == x*q + y*s, a == k*b + c)


    if name == "fermat1_1":
      A, R, u, v, r = Int('A'), Int('R'), Int('u'), Int('v'), Int('r')
      var_names_initial = [A, R, u, v, r]
      A2, R2, u2, v2, r2 = Int('A2'), Int('R2'), Int('u2'), Int('v2'), Int('r2')

      lc = r != 0
      pre = And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
      rec = And(A2 == A, R2 == R, u2 == u, v2 == v, r2 == r)
      post = And(A == ((u + v - 2)/2) * ((u - v)/2))

    if name == "fermat1_2":
      A, R, u, v, r = Int('A'), Int('R'), Int('u'), Int('v'), Int('r')
      var_names_initial = [A, R, u, v, r]
      A2, R2, u2, v2, r2 = Int('A2'), Int('R2'), Int('u2'), Int('v2'), Int('r2')

      lc = r > 0
      pre = And(A >= 3, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
      rec = And(r2 == r - v, v2 == v + 2, A2 == A, R2 == R, u2 == u)
      post = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A>=3)


    if name == "fermat1_3":
      A, R, u, v, r = Int('A'), Int('R'), Int('u'), Int('v'), Int('r')
      var_names_initial = [A, R, u, v, r]
      A2, R2, u2, v2, r2 = Int('A2'), Int('R2'), Int('u2'), Int('v2'), Int('r2')

      lc = r < 0
      pre = And(A >= 3, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
      rec = And(r2 == r + u, u2 == u + 2, A2 == A, R2 == R, v2 == v)
      post = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)


    if name == "fermat2_1":
      A, R, u, v, r = Int('A'), Int('R'), Int('u'), Int('v'), Int('r')
      var_names_initial = [A, R, u, v, r]
      A2, R2, u2, v2, r2 = Int('A2'), Int('R2'), Int('u2'), Int('v2'), Int('r2')

      lc = r != 0
      pre = And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
      rec = And(A2 == A, R2 == R,
              Or(And(r > 0, r2 == r - v, v2 == v + 2, u2 == u),
                  And(r <= 0, r2 == r + u, u2 == u + 2, v2 == v)))
      post = And(A == ((u + v - 2)/2) * ((u - v)/2))

    if name == "freire1_1":
      a, x, r = Int('a'), Int('x'), Int('r')
      var_names_initial = [a, x, r]
      a2, x2, r2 = Int('a2'), Int('x2'), Int('r2')

      lc = x > r
      pre = And(a > 0, r == 0, x == ToReal(a)/2)
      rec = And(x2 == x - r, r2 == r + 1, a2 == a)
      post = And(r * r + r >= a, r * r - r <= a)

    if name == "freire2_data_1":
      x, a, r, s = Int('x'), Int('a'), Int('r'), Int('s')
      var_names_initial = [x, a, r, s]
      x2, a2, r2, s2 = Int('x2'), Int('a2'), Int('r2'), Int('s2')

      lc = ToReal(x) > ToReal(s)
      pre = And(ToReal(a) > 0, r == 1, ToReal(x) == ToReal(a), ToReal(s) == 3.25)
      rec = And(ToReal(x2) == ToReal(x) - ToReal(s), r2 == r + 1, ToReal(s2) == ToReal(s) + 6 * r + 3, ToReal(a2) == ToReal(a))
      post = And( ((4*r*r*r - 6*r*r + 3*r) + (4*x - 4*a)) == 1, (4*s) - 12*r*r == 1, 8*r*s - 24*a + 16*r - 12*s + 24*x - 3 == 0)
      # post = And(True)

    if name == "geo1_data":
      x, y, z, c, k = Int('x'), Int('y'), Int('z'), Int('c'), Int('k')
      var_names_initial = [x, y, z, c, k]
      x2, y2, z2, c2, k2 = Int('x2'), Int('y2'), Int('z2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(x == 1, y == z, c == 1, z >= 0, z <= 10, k >= 0, k <= 10)
      rec = And(c2 == c + 1, k2 == k, z2 == z, x2 == x * z + 1, y2 == y * z)
      post = And(x * (z-1) == y - 1)

    if name == "geo2_1":
      x, y, z, c, k = Int('x'), Int('y'), Int('z'), Int('c'), Int('k')
      var_names_initial = [x, y, z, c, k]
      x2, y2, z2, c2, k2 = Int('x2'), Int('y2'), Int('z2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(x == 1, y == 1, c == 1, z >= 0, z <= 10, k >= 0, k <= 10)
      rec = And(c2 == c + 1, k2 == k, z2 == z, x2 == x * z + 1, y2 == y * z)
      post = And(x * (z - 1) == z * y - 1)

    if name == "geo3_data":
      x, y, z, c, k, a = Int('x'), Int('y'), Int('z'), Int('c'), Int('k'), Int('a')
      var_names_initial = [x, y, z, c, k, a]
      x2, y2, z2, c2, k2, a2 = Int('x2'), Int('y2'), Int('z2'), Int('c2'), Int('k2'), Int('a2')

      lc = c < k
      pre = And(x == a, y == 1, c == 1, z >= 0, z <= 10, k >= 0, k <= 10)
      rec = And(c2 == c + 1, k2 == k, z2 == z, x2 == x * z + a, y2 == y * z, a2 == a)
      post = And(x * z - x + a - a * z * y == 0)

    if name == "hard_1":
      A, B, q, r, d, p = Int('A'), Int('B'), Int('q'), Int('r'), Int('d'), Int('p')
      var_names_initial = [A, B, q, r, d, p]
      A2, B2, q2, r2, d2, p2 = Int('A2'), Int('B2'), Int('q2'), Int('r2'), Int('d2'), Int('p2')

      lc = r >= d
      pre = And(A >= 0, B >= 1, r == A, d == B, p == 1, q == 0)
      rec = And(d2 == 2 * d, p2 == 2 * p, A2 == A, B2 == B, r2 == r, q2 == q)
      post = And(A >= 0, B >= 1, r == A, d == B * p, q == 0, r < d)

    if name == "hard_2":
      A, B, q, r, d, p = Int('A'), Int('B'), Int('q'), Int('r'), Int('d'), Int('p')
      var_names_initial = [A, B, q, r, d, p]
      A2, B2, q2, r2, d2, p2 = Int('A2'), Int('B2'), Int('q2'), Int('r2'), Int('d2'), Int('p2')

      lc = p != 1
      pre = And(A >= 0, B >= 1, r == A, d == B * p, q == 0, r < d)
      rec = And(A2 == A, B2 == B, d2 == d/2, d == 2 * d2, p2 == p/2, p == 2 * p2, Or(And(r >= d2, r2 == r - d2, q2 == q + p2), And(r < d2, r2 == r, q2 == q)))
      temp = Int('temp')
      post = Exists(temp, And(temp >= 0, temp < B, A == q * B + temp))

    if name == "knuth_1":
      a, d, r, n, t, k, q, s = Ints('a d r n t k q s')
      var_names_initial = [a, d, r, n, t, k, q, s]
      a2, d2, r2, n2, t2, k2, q2, s2 = Ints('a2 d2 r2 n2 t2 k2 q2 s2')

      lc = And(s>=d, r!=0)
      pre = And(n % 2 == 1, n > 0, a % 2 == 1,  (a - 1) * (a - 1) * (a - 1) < 8*n, d == a, r == n % d, t == 0, k==n % (d-2), q==4*(n/(d-2) - n/d), s ==  ToInt(n ** (Real(1)/2)))
      rec = And(If(2*r-k+q<0, And(t2 == r, r2==2*r-k+q+d+2, k2==t2, q2==q+4, d2==d+2), If(And(2*r-k+q>=0, 2*r-k+q<d+2), And(t2 == r, r2==2*r-k+q, k2==t2, d2 == d+2, q2 == q), If(And(2*r-k+q>=0, 2*r-k+q>=d+2, 2*r-k+q<2*d+4),And(t2 == r, r2==2*r-k+q-d-2, k2==t2, q2==q-4, d2==d+2), And(t2 == r, r2==2*r-k+q-2*d-4, k2==t2, q2==q-8, d2==d+2)))), a2==a, n2 == n, s2 == s)
      # post = And(d*d*q - 2*q*d - 4*r*d + 4*k*d  + 8*r == 8*n, k*t == t*t, d*d*q - 2*d*q - 4*d*r + 4*d*t + 4*a*k - 4*a*t - 8*n + 8*r == 0, d*k - d*t - a*k + a*t == 0)
      post = And(d*d*q - 2*q*d - 4*r*d + 4*k*d  + 8*r == 8*n, d % 2 == 1)

    if name == "lcm1_1":
      x, y, u, v, a, b = Int('x'), Int('y'), Int('u'), Int('v'), Int('a'), Int('b')
      var_names_initial = [x, y, u, v, a, b]
      x2, y2, u2, v2, a2, b2 = Int('x2'), Int('y2'), Int('u2'), Int('v2'), Int('a2'), Int('b2')

      lc = x != y
      pre = And(a >= 1, b>=1, x == a, y == b, u == b, v == 0)
      rec = And(If(x > y, If(x%y == 0, And(x2 == x - y, v2 == v + u), And(x2 == x - y * (x/y), v2 == v + u*(x/y))),And(x2 == x, v2 == v)), If(x2 < y, If(y%x2 == 0, And(y2 == y - x2 , u2 == u + v2), And(y2 == y - x2 * (y/x2), u2 == u + v2 * (y/x2))), And(y2 == y, v2 == v)), a2 == a, b2 == b)
      post = And(x*u + y*v == a*b)


    if name == "lcm1_2":
      x, y, u, v, a, b = Int('x'), Int('y'), Int('u'), Int('v'), Int('a'), Int('b')
      var_names_initial = [x, y, u, v, a, b]
      x2, y2, u2, v2, a2, b2 = Int('x2'), Int('y2'), Int('u2'), Int('v2'), Int('a2'), Int('b2')

      lc = x > y
      pre = And(x!=y, x*u + y*v == a*b, a>=1, b>=1)
      rec = And(x2 == x - y, v2 == v+u, y2 == y, u2 == u, a2 == a, b2 == b)
      post = And(x*u + y*v == a*b)

    if name == "lcm1_3":
      x, y, u, v, a, b = Int('x'), Int('y'), Int('u'), Int('v'), Int('a'), Int('b')
      var_names_initial = [x, y, u, v, a, b]
      x2, y2, u2, v2, a2, b2 = Int('x2'), Int('y2'), Int('u2'), Int('v2'), Int('a2'), Int('b2')

      lc = x < y
      pre = And(y == b, a>=1, b>=1, u == b, If(a > b, And(x == a - b * (a/b), v == b * (a/b)), And(x == a, v == 0)))
      rec = And(y2 == y - x, u2 == u+v, x2 == x, v2 == v, a2 == a, b2 == b)
      post = And(x*u + y*v == a*b)

    if name == "lcm2_1":
      x, y, u, v, a, b = Int('x'), Int('y'), Int('u'), Int('v'), Int('a'), Int('b')
      var_names_initial = [x, y, u, v, a, b]
      x2, y2, u2, v2, a2, b2 = Int('x2'), Int('y2'), Int('u2'), Int('v2'), Int('a2'), Int('b2')

      lc = x != y
      pre = And(a >= 1, b >= 1, x == a, y == b, u == b, v == a)
      rec = And(a2 == a, b2 == b,
                Or(And(x > y, x2 == x - y, v2 == v + u, u2 == u, y2 == y),
                  And(x <= y, y2 == y - x, u2 == u + v, v2 == v, x2 == x)))
      post = And(x*u + y*v == 2*a*b)


    if name == "mannadiv_1":
      A, B, q, r, t = Int('A'), Int('B'), Int('q'), Int('r'), Int('t')
      var_names_initial = [A, B, q, r, t]
      A2, B2, q2, r2, t2 = Int('A2'), Int('B2'), Int('q2'), Int('r2'), Int('t2')

      lc = t != 0
      pre = And(A >= 0, B >= 1, q == 0, r == 0, t == A)
      rec = And(A2 == A, B2 == B,
                Or(And(r + 1 == B, q2 == q + 1, r2 == 0, t2 == t - 1),
                    And(r + 1 != B, q2 == q, r2 == r + 1, t2 == t - 1)))
      temp = Int('temp')
      post = Exists(temp, And(A == q * B + temp, temp >= 0, temp < B))

    if name == "prod4br_1":
      x, y, a, b, p, q = Int('x'), Int('y'), Int('a'), Int('b'), Int('p'), Int('q')
      var_names_initial = [x, y, a, b, p, q]
      x2, y2, a2, b2, p2, q2 = Int('x2'), Int('y2'), Int('a2'), Int('b2'), Int('p2'), Int('q2')

      lc = And(a != 0, b != 0)
      pre = And(x >= 1, y >= 1, x == a, y == b, p == 1, q == 0)
      rec = And(x2 == x, y2 == y,
                Or(And(a % 2 == 0, b % 2 == 0, a2 == a/2, b2 == b/2, p2 == 4 * p, q2 == q),
                    And(a % 2 == 1, b % 2 == 0, a2 == a - 1, b2 == b, p2 == p, q2 == q + b * p),
                    And(a % 2 == 0, b % 2 == 1, a2 == a, b2 == b - 1, p2 == p, q2 == q + a * p),
                    And(a % 2 == 1, b % 2 == 1, a2 == a - 1, b2 == b - 1, p2 == p, q2 == q + (a2 + b2 + 1) * p)))
      post = And(q == x * y)

    if name == "prodbin_1":
      a, b, x, y, z = Int('a'), Int('b'), Int('x'), Int('y'), Int('z')
      var_names_initial = [a, b, x, y, z]
      a2, b2, x2, y2, z2 = Int('a2'), Int('b2'), Int('x2'), Int('y2'), Int('z2')

      lc = y != 0
      pre = And(a >= 0, b >= 0, x == a, y == b, z == 0)
      rec = And(a2 == a, b2 == b,
                Or(And(y % 2 == 1, z2 == z + x, y2 == (y - 1)/2, x2 == 2 * x),
                    And(y % 2 == 0, z2 == z, y2 == y/2, x2 == 2 * x)))
      post = And(z == a * b)

    if name == "ps2_1":
      var_names_initial = [Int('x'), Int('y'), Int('c'), Int('k')]
      x, y, c, k = Int('x'), Int('y'), Int('c'), Int('k')
      x2, y2, c2, k2 = Int('x2'), Int('y2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
      rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2)
      post = And(2 * x - k * k - k == 0)


    if name == "ps3_1":
      var_names_initial = [Int('x'), Int('y'), Int('c'), Int('k')]
      x, y, c, k = Int('x'), Int('y'), Int('c'), Int('k')
      x2, y2, c2, k2 = Int('x2'), Int('y2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
      rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2 * y2)
      post = And(6 * x - 2 * k * k * k - 3 * k * k - k == 0)


    if name == "ps4_1":
      var_names_initial = [Int('x'), Int('y'), Int('c'), Int('k')]
      x, y, c, k = Int('x'), Int('y'), Int('c'), Int('k')
      x2, y2, c2, k2 = Int('x2'), Int('y2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
      rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2 * y2 * y2)
      post = And(4 * x - k * k * k * k - 2 * k * k * k - k * k == 0)

    if name == "ps5_data":
      var_names_initial = [Int('x'), Int('y'), Int('c'), Int('k')]
      x, y, c, k = Int('x'), Int('y'), Int('c'), Int('k')
      x2, y2, c2, k2 = Int('x2'), Int('y2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
      rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2 * y2 * y2 * y2)
      post = And(6 * k * k * k * k * k + 15 * k * k * k * k + 10 * k * k * k - 30 * x - k == 0)


    if name == "ps6_1":
      var_names_initial = [Int('x'), Int('y'), Int('c'), Int('k')]
      x, y, c, k = Int('x'), Int('y'), Int('c'), Int('k')
      x2, y2, c2, k2 = Int('x2'), Int('y2'), Int('c2'), Int('k2')

      lc = c < k
      pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
      rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2 * y2 * y2 * y2 * y2)
      post = And(-2 * k * k * k * k * k * k - 6 * k * k * k * k * k -5 * k * k * k * k + k * k + 12 * x == 0)

    if name == "sqrt1_1_data":
      var_names_initial = [Int('a'), Int('n'), Int('t'), Int('s')]
      a, n, t, s = Int('a'), Int('n'), Int('t'), Int('s')
      a2, n2, t2, s2 = Int('a2'), Int('n2'), Int('t2'), Int('s2')

      lc = s <= n
      pre = And(n >= 0, a == 0, s == 1, t == 1)
      rec = And(n2 == n, a2 == a + 1, t2 == t + 2, s2 == s + t2)
      post = And(t*t - 4*s + 2*t + 1 == 0, (a+1) * (a+1) == s, t == 2 * a + 1 )

    return pre, rec, post, lc, var_names_initial
  else:
    global args
    
    functions = {"And": z3.And, "Or": z3.Or}
    vars_dict = {str(var): var for var in var_names_initial}
    vars_dict['And'] = z3.And
    vars_dict['Or'] = z3.Or
    vars_dict['unknown'] = unknown
    vars_dict.update({str(var)+'2': Int(str(var)+'2') for var in var_names_initial})
    
    return eval(args.pre_condition, vars_dict), eval(args.inductive_condition, vars_dict), eval(args.post_condition, vars_dict), eval(args.loop_condition,vars_dict), var_names_initial

#Checking with z3
def check_adequacy(pre, lc, rec, post, invariant, var_names_initial,is_timeout = False):
  solver = z3.Solver()
  solver.set('random_seed', 5)
  subs = [(Int(str(var)),Int(str(var)+'2')) for var in var_names_initial]
  #print(subs)
  invariant2 = z3.substitute(invariant, subs)
  
  pre_ret = ["proved"]
  rec_ret = []
  post_ret = []
  unsat_count = 1

  
  attempt = 0

  while(attempt < 10):
    
    solver.set("timeout",40000)
    
    #solver = z3.Solver()
    if str(lc) == "unknown":
      solver.add(Not(Implies(And(invariant, rec), invariant2)))
    else:
      solver.add(Not(Implies(And(invariant, lc, rec), invariant2)))
    
    result = solver.check()

    if result == unsat:
      rec_ret = ["proved"]
      unsat_count += 1
      break
    elif result == sat:
      rec_ret = solver.model()
      break
    else:
      rec_ret = ["unknown"]
      attempt +=1

    solver.reset()
  solver.reset()
  #solver = z3.Solver()
  solver.set("timeout",15000)
  # if is_timeout == True:
  #   solver.set("timeout",1000)
  if str(lc) == "unknown":
    solver.add(Not(Implies(And(invariant), post)))
  else:  
    solver.add(Not(Implies(And(invariant, Not(lc)), post)))
  result = solver.check()
  
  if result == unsat:
    post_ret = ["proved"]
    unsat_count += 1
  elif result == sat:
    post_ret = solver.model()
  else:
    post_rec = ["unknown"]
  solver.reset()
  
  return pre_ret, rec_ret, post_ret

def perform_adequacy_check(invariant, min_max_invariants, name, bounds):
  solver = z3.Solver()
  
  solver.set('random_seed', 5)
  invariant_list = invariant.copy()
  # print(bounds)
  # bounds = []
  #invariant_list.extend(min_max_invariants)
  pre, rec, post, lc, var_names_initial = pre_rec_post_conditions(name)
  # bounds.extend(generate_2d_inequality(X_poly, columns))
  # bounds.extend(generate_3d_inequality(X_poly, columns))

  counter = 0
  while counter < 1:    
    
    satisfy_one_inv = []
    solver.set('timeout',15000)
    for inv in invariant_list:
      solver.add(Not(Implies(pre, inv)))
      if solver.check() == unsat:
        satisfy_one_inv.append(inv)
      
      solver.reset()
    
    satisfy_one_bounds = []
    for bound in bounds:
      solver.add(Not(Implies(pre, bound)))
      if solver.check() == unsat:
        satisfy_one_bounds.append(bound)
      
      solver.reset()
    
    min_max_inv = []
    for inv in min_max_invariants:
      solver.add(Not(Implies(pre, inv)))
      if solver.check() == unsat:
        min_max_inv.append(inv)
      
      solver.reset()
    min_max_invariants = min_max_inv.copy()
    
    solver.set('timeout',15000)

    
    if str(lc) == "unknown":
      solver.add(Not(Implies(And(simplify(And(satisfy_one_inv+satisfy_one_bounds+min_max_invariants))), simplify(post))))
    else:
      solver.add(Not(Implies(And(simplify(And(satisfy_one_inv+satisfy_one_bounds+min_max_invariants)),Not(lc)), simplify(post))))

    
    ans3 = solver.check()
    solver.reset()
    
    if ans3 == unsat:
      ans3 = ["proved"]
    else:
      ans3 = ["error"]
    solver.set('timeout',-1)
    
    ans1 = ["error"]
    ans2 = ["error"]
    if ans3 != ["proved"]:
      ans1 = ['error']
      ans2 = ['error']
      ans3 = ['error']
      invariant = "error"
      counter += 1
      continue

    satisfy_one = []

    solver.set("timeout",5000)
    subs = [(Int(str(var)),Int(str(var)+'2')) for var in var_names_initial]
    
    for _ in range(len(satisfy_one_inv)+len(min_max_invariants)):
      solver.push()
      min_max_invariant2 = z3.substitute(And(min_max_invariants), subs)
      satisfy_one_inv2 = z3.substitute(And(satisfy_one_inv), subs)
      if str(lc) != "unknown":
        solver.add(Not(Implies(And(And(satisfy_one_inv+min_max_invariants),rec, lc),And(min_max_invariant2, satisfy_one_inv2))))
      else:
        solver.add(Not(Implies(And(And(satisfy_one_inv+min_max_invariants),rec),And(min_max_invariant2, satisfy_one_inv2))))
      res = solver.check()
      
      if res == unsat:
        satisfy_one_inv.extend(min_max_invariants)
        solver.pop()
        break
      elif res == unknown:
        solver.pop()
        pass
      else:
        counter_example = solver.model()
        solver.pop()
        implication_ct = [Int(str(ct)) == counter_example[ct] for ct in counter_example if str(ct) != 'div0' and str(ct) != "mod0"]
        updated_min_max = []
        
        for inv in min_max_invariants:
          solver.push()
          solver.add(Not(Implies(And(implication_ct), z3.substitute(inv, subs))))
          result = solver.check()
          if result == unsat:
            updated_min_max.append(inv)
          solver.pop()
        min_max_invariants = updated_min_max.copy()
    
    valid_bounds = []
    valid_bound2s = []
    solver.set("timeout", 5000)
    prev_len = -1
    satisfy_one_inv2 = z3.substitute(And(satisfy_one_inv), subs).children()
    satisfy_one = satisfy_one_inv.copy()
    
    for bound in list(set(satisfy_one_bounds).difference(set(valid_bounds))):
      if str(bound) == "True":
        continue
      bound2 = z3.substitute(bound, subs)
      
      attempt = 0
      while attempt < 7:

        solver.push()
        if str(lc) != "unknown":
          solver.add(Not(Implies(And(And([bound]+satisfy_one_inv),lc, rec), And([bound2]+satisfy_one_inv2))))
        else:
          solver.add(Not(Implies(And(And([bound]+satisfy_one_inv), rec), And([bound2]+satisfy_one_inv2))))

        result = solver.check()
        if result == z3.unsat:
            valid_bounds.append(bound)
            valid_bound2s.append(bound2)
            solver.pop()
            break
        elif result == unknown:
          attempt += 1
        else:
          solver.pop()
          break
        solver.pop()
      
    remaining_bounds = list(set(satisfy_one_bounds).difference(set(valid_bounds)))
    is_inductive = 0
    solver.set("timeout",5000)
    for _ in range(len(remaining_bounds)):
      
      remaining_bound2s = z3.substitute(And(remaining_bounds), subs).children()
      valid_bound2s = z3.substitute(And(valid_bounds), subs).children()
      solver.push()
      if str(lc) != "unknown":
        solver.add(Not(Implies(And(And(valid_bounds+remaining_bounds),lc, rec), And(valid_bound2s+remaining_bound2s))))
      else:
        solver.add(Not(Implies(And(And(valid_bounds+remaining_bounds), rec), And(valid_bound2s+remaining_bound2s))))
      
      result = solver.check()
      if result == unsat:
        solver.pop()
        is_inductive = 1
        break
      elif result == unknown:
        solver.pop()
        continue
      else:
        counter_example = solver.model()
        implication_ct = [Int(str(ct)) == counter_example[ct] for ct in counter_example if str(ct) != 'div0' and str(ct) != "mod0"]
        solver.pop()

        for bound in remaining_bounds:

          if str(bound) == 'True':
            remaining_bounds.remove(bound)
            continue
          solver.push()
          bound2 = z3.substitute(bound, subs)
          
          solver.add(Not(Implies(And(*implication_ct), bound2)))
          
          result = solver.check()
          if result == sat:
            remaining_bounds.remove(bound)
          solver.pop()
        
    if is_inductive == 1:
      valid_bounds.extend(remaining_bounds)
    satisfy_one.extend(valid_bounds)
    
    # print(satisfy_one)
    solver.set("timeout", -1)
    invariant = satisfy_one.copy()
    if ans3 !=['error']:
      if len(valid_bounds) == 0:
        ans1, ans2, ans3 = check_adequacy(pre, lc, rec, post,simplify(And(invariant)), var_names_initial)
        if ans1 == ["proved"] and ans2 == ["proved"] and ans3 == ["proved"]:
          return ans1, ans2, ans3, invariant
      else:
        solver.set('timeout',50000)

    
        if str(lc) == "unknown":
          solver.add(Not(Implies(And(simplify(And(invariant))), simplify(post))))
        else:
          solver.add(Not(Implies(And(simplify(And(invariant)),Not(lc)), simplify(post))))

        
        ans3 = solver.check()
        solver.reset()
        if ans3 == unsat:
          return ["proved"],["proved"], ["proved"], invariant      
    counter += 1    
  return ans1, ans2, ans3, invariant

def generate_degrees(deg, X, simply_equal_constraints, columns):
  poly = PolynomialFeatures(degree=deg)
  if X.shape[1] >= 1:
    X_poly = poly.fit_transform(X)
    X_poly = X_poly[:,1:]
    total_vars = poly.get_feature_names_out(columns.values)
    total_vars = total_vars[1:]
  else:
    X_poly = X
    total_vars = []
  
  total_variables_processed = []
  for variable in total_vars:
    temps = []
    for var in variable.split():
      if len(var.split("^")) > 1:
        temps.extend([var.split("^")[0]] * int(var.split("^")[-1]))
      else:
        temps.extend([var])
    total_variables_processed.append(temps.copy())

  z3_vars = []
  for variable in total_variables_processed:
    var_to_append = Int(variable[0])
    for var in variable[1:]:
      var_to_append = var_to_append * Int(var) 
    z3_vars.append(var_to_append)

  min_max_invariants = []
  for col in range(len(z3_vars)):
    if '*' not in str(z3_vars[col]) and min(X_poly[:,col]) >= -100000:
      min_max_invariants.append(z3_vars[col] >= min(X_poly[:,col]))
    if '*' not in str(z3_vars[col]) and max(X_poly[:,col]) <= 100000:
      min_max_invariants.append(z3_vars[col] <= max(X_poly[:,col]))
    if '*' in str(z3_vars[col]) and len(str(z3_vars[col]).split('*')) == 2:
      col1, col2 = list(total_vars).index(str(z3_vars[col]).split('*')[0]), list(total_vars).index(str(z3_vars[col]).split('*')[1])
      if col1 != col2 and min(X_poly[:,col1]) < 0 and min(X_poly[:,col2]) < 0 and np.all(X_poly[:,col1] * X_poly[:,col2] >=0):
        min_max_invariants.append(z3_vars[col] >= min(X_poly[:,col]))
  

  equals = {}
  for data_index, data in enumerate(X_poly.T):
    data_hash = str(hash(str(list(data))))
    if data_hash in equals:
      equals[data_hash].append(data_index)
    else:
      equals[data_hash] = [data_index]

  grouped_cols = list(equals.values())
  cols_to_be_droped = []
  for index, equal_cols in enumerate(grouped_cols):
    if len(equal_cols) > 1:
      for var in equal_cols[1:]:
        simply_equal_constraints.append(z3_vars[equal_cols[0]] == z3_vars[var])
      cols_to_be_droped.extend(equal_cols[1:])
  
  X_poly = np.delete(X_poly, cols_to_be_droped, axis = 1)
  total_vars = list(np.delete(np.array(total_vars), cols_to_be_droped, axis = 0))
  z3_vars = list(np.delete(np.array(z3_vars), cols_to_be_droped, axis = 0))
  
  return X_poly, min_max_invariants, z3_vars, simply_equal_constraints

invs_before_houdini = {}
def run(file_path, deg):
  global var_names_initial
  start_time_before_houdini = time.time()
  name = file_path[-1::-1]
  name = name[4:name.index('/')][-1::-1]
  datas = pd.read_csv(file_path)

  simply_equal_constraints = []

  var_names = [Int(var) for var in list(datas.columns[:])]
  var_names_initial = var_names.copy()

  equals = {}
  new_data = datas.iloc[:,:]

  for data_index, data in enumerate(new_data.T.values):
    data_hash = str(hash(str(list(data))))
    
    if data_hash in equals:
      equals[data_hash].append(data_index)
    else:
      equals[data_hash] = [data_index]

  grouped_cols = list(equals.values())

  cols_to_be_droped = []
  for index, equal_cols in enumerate(grouped_cols):
    if len(equal_cols) > 1:
      for var in equal_cols[1:]:
        simply_equal_constraints.append(var_names[equal_cols[0]] == var_names[var])
      cols_to_be_droped.extend(equal_cols[1:])

  new_data.drop(new_data.iloc[:,cols_to_be_droped], inplace = True, axis = 1)
  var_names = list(np.delete(np.array(var_names), cols_to_be_droped, axis = 0))

  constants = new_data[[i for i in new_data if new_data[i].nunique()==1]] 
  new_data = new_data[[i for i in new_data if new_data[i].nunique()>1]]
  for col in list(constants.columns):
    simply_equal_constraints.append(ToReal(Int(col)) == constants[col][0])

  X = new_data.values
  if X.shape[1] >= 1:
    y = new_data.iloc[:, 0].values

  # #---------------Bitwise And------------------------------
  bitwise_and = []
  for i in range(len(new_data.columns)):
    if min(new_data.iloc[:,i]) != 0 and new_data.iloc[:,i].dtype == int and min(new_data.iloc[:,i]) <= 10:
      for num in range(1,min(new_data.iloc[:,i]) + 1):
        if(np.all(new_data.iloc[:,i] & num == new_data.iloc[0,i] & num)):
          number = int(new_data.iloc[0,i] & num)
          bitwise_and.append(Int2BV(Int(new_data.columns[i]), 16) & num == number)

  #---------------Mod operation-------------------------------------
  for i in range(len(new_data.columns)):
    col = new_data.columns[i]
    
    for num in range(2, 11):
      if len(set(new_data.iloc[:,i] % num)) == 1:
        simply_equal_constraints.append(Int(col) % num == int(new_data.iloc[0,i]%num))
    
    
    min_element = int(min(new_data.iloc[:,i].to_list()))
    if min_element > 10:
      elements = [ele for ele in new_data.iloc[:,i].to_list() if int(ele) != min_element]

      min2 = int(min(elements))
      mod_val = min2 % min_element

      mods = set(np.mod(new_data.iloc[:,i].to_list(), min_element + mod_val))
      
      if len(mods) == 1:
        simply_equal_constraints.append(Int(col) % (min_element + mod_val) == min(min_element, min_element * mod_val))



  cols_to_append = {}

  for i in range(len(new_data.columns)-1):
    if(np.all(new_data.iloc[:,i].astype(int) == new_data.iloc[:,i])):
      for j in range(i+1,len(new_data.columns)):
        if(np.all(new_data.iloc[:,j].astype(int) == new_data.iloc[:,j])):
          if str(np.gcd(new_data.iloc[:,i].astype(int), new_data.iloc[:,j].astype(int))) in cols_to_append:
            cols_to_append[str(np.gcd(new_data.iloc[:,i].astype(int), new_data.iloc[:,j].astype(int)))].append("gcd("+str(new_data.columns[i])+","+str(new_data.columns[j])+")")
          else:
            cols_to_append[str(np.gcd(new_data.iloc[:,i].astype(int), new_data.iloc[:,j].astype(int)))] = ["gcd("+str(new_data.columns[i])+","+str(new_data.columns[j])+")"]
  
  gcd_invs = []
  for gcds in list(cols_to_append.values()):
    if len(gcds) == 1:
      continue
    for gcd_pair1 in range(len(gcds)-1):
      for gcd_pair2 in range(gcd_pair1+1,len(gcds)):
        gcd_invs.append(gcds[gcd_pair1] + '=='+ gcds[gcd_pair2])


  X_poly, min_max_invariants, z3_vars, simply_equal_constraints = generate_degrees(deg, X, simply_equal_constraints, new_data.columns)
  
  equality_invariant_list, final_coefs, final_intercepts = equality_invariants(X_poly)

  invariant = post_process_equality_simple_invariant(equality_invariant_list, final_coefs, final_intercepts, simply_equal_constraints, z3_vars, name)

  bounds = generate_2d_inequality(X_poly, new_data.columns)

  bounds.extend(generate_3d_inequality(X_poly, new_data.columns))

  end_time_before_houdini = time.time()

  invs_before_houdini[name] = len(invariant+min_max_invariants+bounds+gcd_invs+bitwise_and)
  ans1, ans2, ans3, invariant = perform_adequacy_check(invariant, min_max_invariants, name, bounds)

  end_time_after_houdini = time.time()

  if ans1 == ['proved'] and ans2 == ['proved'] and ans3 == ['proved']:
    return simplify(And(*invariant)), gcd_invs, bitwise_and, end_time_before_houdini - start_time_before_houdini,  end_time_after_houdini - end_time_before_houdini
  else:
    return "Adequate invariant doesn't exists", gcd_invs, bitwise_and, end_time_before_houdini - start_time_before_houdini,  end_time_after_houdini - end_time_before_houdini

import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--benchmark", help="Benchmark name: nla/zilu/loop-inv")
parser.add_argument("-p","--folder_path",help="Mention the path of the csv file", default = '.')
parser.add_argument("-fn","--filename", help = "Mention filename")
parser.add_argument('-pre','--pre_condition', help = "Specify preconditions in z3 format ex: And(x >= y)")
parser.add_argument('-ind','--inductive_condition', help = 'Specify inductive condition in z3 format ex: x >= y')
parser.add_argument('-post','--post_condition', help = 'Specify Post condition in z3 format ex. And(x >= y)', default = 'True')
parser.add_argument('-lc','--loop_condition', help='Specify loop condition as per z3 format ex. And(x>=y). If non-deterministic write unknown', default = 'unknown')
parser.add_argument('-degree','--degree', help="Mention the degree upto which non-linear terms are to be generated", default = 2)


args = parser.parse_args()

external_file = 0
var_names_initial = []

nla_degrees = {'cohencu_data.csv': 3, 'cohendiv_1.csv': 2, 'cohendiv_2.csv': 2, 'dijkstra_1.csv': 1, 'dijkstra_2.csv': 2, 'divbin_1.csv': 1, 'divbin_2.csv': 2, 'egcd2_1.csv': 2, 'egcd2_2_data.csv': 2, 'egcd3_1_data.csv': 2, 'egcd3_2_data.csv': 2, 'egcd3_3_data.csv': 2, 'egcd_1_data.csv': 2, 'fermat1_1.csv': 2, 'fermat1_2.csv': 2, 'fermat1_3.csv': 2, 'fermat2_1.csv': 2, 'freire1_1.csv': 2, 'freire2_data_1.csv': 4, 'geo1_data.csv': 2, 'geo2_1.csv': 2, 'geo3_data.csv': 3, 'hard_1.csv': 2, 'hard_2.csv': 2, 'knuth_1.csv': 3, 'lcm1_1.csv': 2, 'lcm1_2.csv': 2, 'lcm1_3.csv': 2, 'lcm2_1.csv': 2, 'mannadiv_1.csv': 2, 'prod4br_1.csv': 3, 'prodbin_1.csv': 2, 'ps2_1.csv': 2, 'ps3_1.csv': 3, 'ps4_1.csv': 4, 'ps5_data.csv': 5, 'ps6_1.csv': 6, 'sqrt1_1_data.csv': 2}

zilu_degrees = {'benchmark01_conjunctive.csv': 1, 'benchmark02_linear.csv': 1, 'benchmark03_linear.csv': 1, 'benchmark04_conjunctive.csv': 1, 'benchmark05_conjunctive.csv': 1, 'benchmark06_conjunctive.csv': 1, 'benchmark07_linear.csv': 1, 'benchmark08_conjunctive.csv': 1, 'benchmark09_conjunctive.csv': 1, 'benchmark10_conjunctive.csv': 1, 'benchmark11_linear.csv': 1, 'benchmark12_linear.csv': 1, 'benchmark13_conjunctive.csv': 1, 'benchmark14_linear.csv': 1, 'benchmark15_conjunctive.csv': 1, 'benchmark16_conjunctive.csv': 1, 'benchmark17_conjunctive.csv': 1, 'benchmark18_conjunctive.csv': 1, 'benchmark19_conjunctive.csv': 1, 'benchmark20_conjunctive.csv': 1, 'benchmark21_disjunctive.csv': 1, 'benchmark22_conjunctive.csv': 1, 'benchmark23_conjunctive.csv': 1, 'benchmark24_conjunctive.csv': 1, 'benchmark25_linear.csv': 1, 'benchmark26_linear.csv': 1, 'benchmark27_linear.csv': 1, 'benchmark28_linear.csv': 1, 'benchmark29_linear.csv': 1, 'benchmark30_conjunctive.csv': 1, 'benchmark31_disjunctive.csv': 1, 'benchmark32_linear.csv': 1, 'benchmark33_linear.csv': 1, 'benchmark34_conjunctive.csv': 1, 'benchmark35_linear.csv': 1, 'benchmark36_conjunctive.csv': 1, 'benchmark37_conjunctive.csv': 1, 'benchmark38_conjunctive.csv': 1, 'benchmark39_conjunctive.csv': 1, 'benchmark40_polynomial.csv': 2, 'benchmark41_conjunctive.csv': 1, 'benchmark42_conjunctive.csv': 1, 'benchmark43_conjunctive.csv': 1, 'benchmark44_disjunctive.csv': 1, 'benchmark45_disjunctive.csv': 1, 'benchmark46_disjunctive.csv': 1, 'benchmark47_linear.csv': 1, 'benchmark48_linear.csv': 1, 'benchmark49_linear.csv': 1, 'benchmark50_linear.csv': 1, 'benchmark51_polynomial.csv': 1, 'benchmark52_polynomial.csv': 1, 'benchmark53_polynomial.csv': 2}

svcomp_loop_inv = {'bin-suffix-5.csv': 1, 'const.csv': 1, 'eq1.csv': 1, 'eq2.csv': 1, 'even.csv': 1, 'linear-inequality-inv-a.csv': 1, 'linear-inequality-inv-b.csv': 1, 'mod4.csv': 1, 'odd.csv': 1}

if args.benchmark == 'loop-inv':
  degrees = svcomp_loop_inv
  args.folder_path = '/benchmark/SV-Comp_loop_invariants_csv'

elif args.benchmark == 'nla':
  degrees = nla_degrees
  args.folder_path = '/benchmark/NLA Benchmark'

elif args.benchmark == 'zilu':
  degrees = zilu_degrees
  args.folder_path = '/benchmark/SVcomp-loop-zilu-csv'

elif args.benchmark != None:
  raise Exception("Give proper benchmark name: nla/zilu/loop-inv")

not_staisfies = [] 
index = 0 

if args.benchmark != None and args.filename != None:
  if not os.path.isfile(os.path.join(args.folder_path, args.filename)) or '.csv' not in args.filename:
    raise Exception("Provide valid path with -p and valid filename with -fn")
  print(args.filename)
  invariant_generated, gcd_invs, bitwise_and, before_houdini, houdini = run(os.path.join(args.folder_path, args.filename), degrees[args.filename])  
  print('~~~~~~~~~Invariant:{}~~~~~~~~~~~'.format(invariant_generated))
  print('~~~~~~~~~GCD:{}~~~~~~~~~~~'.format(gcd_invs))
  print('~~~~~~~~~Bitwise And:{}~~~~~~~~~~~'.format(bitwise_and))

elif args.benchmark != None and args.filename == None:
  for filename in os.listdir(args.folder_path): 
    if os.path.isfile(os.path.join(args.folder_path, filename)):
      print(filename)
      if 'abstracted' in filename:
        continue
      avg_time_before_houdini = 0.0
      avg_time_houdini = 0.0

      satisfy = True
      for _ in range(1):
        invariant_generated, gcd_invs, bitwise_and, before_houdini, houdini = run(os.path.join(args.folder_path, filename), degrees[filename])

        print('~~~~~~~~~Invariant:{}~~~~~~~~~~~'.format(invariant_generated))
        print('~~~~~~~~~GCD:{}~~~~~~~~~~~'.format(gcd_invs))
        print('~~~~~~~~~Bitwise And:{}~~~~~~~~~~~'.format(bitwise_and))

        avg_time_before_houdini += before_houdini
        avg_time_houdini += houdini

        if str(invariant_generated) == "Adequate invariant doesn't exists":
          satisfy = False
          not_staisfies.append(filename)

elif args.benchmark == None and args.filename == None:
  raise Exception("Invalid filename. Either provide benchmark name with -b flag or file name with -fn flag and folder name with -p flag")

elif args.benchmark == None:
  if not os.path.isfile(os.path.join(args.folder_path, args.filename)) or '.csv' not in args.filename:
    raise Exception("Provide valid path with -p and valid filename with -fn")
  else:
    external_file = 1
    print(args.filename)
  if args.pre_condition != None and args.inductive_condition != None : 
    invariant_generated, gcd_invs, bitwise_and, before_houdini, houdini = run(os.path.join(args.folder_path, args.filename), args.degree)  
    print('~~~~~~~~~Invariant:{}~~~~~~~~~~~'.format(invariant_generated))
    print('~~~~~~~~~GCD:{}~~~~~~~~~~~'.format(gcd_invs))
    print('~~~~~~~~~Bitwise And:{}~~~~~~~~~~~'.format(bitwise_and))
  else:
    raise Exception("Write valid pre and inductive cnditions with -pre and -ind flag")


