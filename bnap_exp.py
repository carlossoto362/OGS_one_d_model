import matplotlib.pyplot as plt
import numpy as np

def b_nap_1(NAP,lambda_,e_nap = 0.02875, f_nap = 0.5):
	return e_nap * (1/lambda_)**(f_nap) * NAP

def b_nap_2(NAP,lambda_,e_nap = 0.02875, f_nap = 0.5):
	b_555 = 0.416*NAP**(f_nap)
	v = np.empty(len(NAP))
	for i in range(len(NAP)):
		if NAP[i] < 2:	
			v[i] = 0
		else:
			v[i] = 0.416*(np.log(NAP[i])/np.log(10) - 0.3)
		
	bbp = (e_nap + 0.01*(0.5 - 0.25 * np.log(NAP)/np.log(10) ) * (lambda_/550)**v ) * b_555
	return bbp


NAPs = np.linspace(0.01,5,100)
lambdas = np.linspace(412,700,100)
plt.plot(NAPs,b_nap_1(NAPs,490),label='1')
plt.plot(NAPs,b_nap_2(NAPs,490,0.002,0.766),label='2')
plt.legend()
plt.show()  

