import numpy as np
#import scipy.linalg as sy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import datetime

import sys

sys.path.append('/home/radhit/Downloads/dota2Dataset/');

import mltools as ml

np.random.seed(0)

# load dota2Dataset
X = np.genfromtxt("../data/dota2Train.csv", delimiter=",") # load train dataset
Y = np.genfromtxt("../data/dota2Test.csv", delimiter=",")  # load test dataset
X_feat, X_label = X[:,1:], X[:,0]
Y_feat, Y_label = Y[:,1:], Y[:,0]
X_cid, X_gm, X_gt = X[:,1], X[:,2], X[:,3]
X_heros, Y_heros = X[:,4:], Y[:,4:]


# global variables
num_heros = 113
num_top_heros = 20
num_clust = 5
num_gm = 5
num_gt = 3
num_features = 20
print(X.shape, Y.shape)

# data exploration
clust_id, clust_id_cts = np.unique(X[:,1], return_counts=True)
game_mod, game_mod_cts = np.unique(X[:,2], return_counts=True)
game_type, game_type_cts = np.unique(X[:,3], return_counts=True)

# print values function for debugging
def print_values(arg1, arg2, arg3, val1, val2, arg4=None, val3=None):
	print(arg1)
	print(arg2, val1)
	print(arg3, val2)
	if arg4 is not None:
		print(arg4, val3) 
	print("-------------------------------------------")

# attribute labelling function
def plot_attributes(arg1, arg2, arg3):
	plt.xlabel(arg1)
	plt.ylabel(arg2)
	plt.title(arg3)
	plt.legend();
	plt.show()  	

print_values("Cluster ID Information:", "cluster_id = ", "counts = ", clust_id, clust_id_cts, "vector length = ", len(clust_id))
print_values("Game Mode Information:", "game_mode = ", "counts = ", game_mod, game_mod_cts)
print_values("Game Type Information:", "game_type = ", "counts = ", game_type, game_type_cts)


# computing the top heros according to win percentage
X_heros = (X_heros.T * X_label).T

heros_win_ct = np.count_nonzero(X_heros == 1, axis=0)
heros_loss_ct = np.count_nonzero(X_heros == -1, axis=0)
heros = np.array(*[range(num_heros)]) + 1
	
plt.plot(heros, heros_win_ct, 'r', label='Win Frequency') 
plt.plot(heros, heros_loss_ct, 'g', label='Loss Frequency') 
plot_attributes("Hero Identifier", "Frequency", "Heros v/s Frequency of Occurrence")  	

heros_win_percent = heros_win_ct/(heros_win_ct + heros_loss_ct)
heros_idx = np.argsort(heros_win_ct)
heros_idx = np.flip(heros_idx)
#print(heros_idx)

plt.bar(heros, heros_win_percent, label='Win Percentage') 
plot_attributes("Hero Identifier", "Percentage", "Heros v/s Win Percentage")

# creating top heros feature matrix by stacking individual columns of original matrix
heros_idx = np.sort(heros_idx[:num_top_heros])
X_heros_top = X_heros[:,heros_idx[0]]
X_heros_top = X_heros_top[:,np.newaxis]

for i in range(num_top_heros-1):
	temp_col = X_heros[:,heros_idx[i+1]]
	temp_col = temp_col[:,np.newaxis]
	X_heros_top = np.append(X_heros_top, temp_col, axis=1)
	print(X_heros_top.shape) 
		

# visualizes depedence between some attributes
def dependence_computation(arg1, arg2, arg3, arg4, arg5):
	temp_idx = np.argsort(arg2)
	temp_idx = np.flip(temp_idx)

	for i in range(arg1):
		temp = X_heros_top[arg4==arg3[temp_idx[i]],:]
		temp_win_ct = np.count_nonzero(temp == 1, axis=0)
		temp_loss_ct = np.count_nonzero(temp == -1, axis=0)
		temp_win_percent = temp_win_ct/(temp_win_ct + temp_loss_ct)
		plt.bar(heros_idx[:num_top_heros], temp_win_percent, label= arg5 + ": "+str(arg3[temp_idx[i]]))
		#plot_attributes("Hero Identifier", "Win Percentage", "Heros v/s Win Percentage (across " + arg5 + ")")


dependence_computation(num_clust, clust_id_cts, clust_id, X_cid, "Cluster ID")
plt.xlabel("Hero Identifier")
plt.ylabel("Win Percentage")
plt.title("Heros v/s Win Percentage (across Cluster ID)")
plt.legend();  
plt.show()
dependence_computation(num_gm, game_mod_cts, game_mod, X_gm, "Game Mode")
plt.xlabel("Hero Identifier")
plt.ylabel("Win Percentage")
plt.title("Heros v/s Win Percentage (across Game Mode)")
plt.legend();
plt.show()
dependence_computation(num_gt, game_type_cts, game_type, X_gt, "Game Type")
plt.xlabel("Hero Identifier")
plt.ylabel("Win Percentage")
plt.title("Heros v/s Win Percentage (across Game Type)")
plt.legend();
plt.show()

def select_features(X, n=num_features):
	#print("Select features with PCA: n=%d" % n)
	pca = PCA(n_components=n)
	pca.fit(standardize(X))
	n_pcs = pca.components_.shape[0]
	most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
	most_important = np.unique(most_important)
	print(most_important)
	return X[:, most_important], most_important

def train_on_SVM(reg):
	print("Training on SVM")
	print("Start Time: ", datetime.datetime.now().ctime())
	clf = make_pipeline(StandardScaler(), LinearSVC(C=reg, random_state=0, tol=1e-5, max_iter=1000))
	clf.fit(X_feat, X_label)
	scores = cross_val_score(clf, X_feat, X_label, cv=10)
	print("Finish Time: ", datetime.datetime.now().ctime())
	print(scores)

	Y_pred = clf.predict(Y_feat)
	test_acc = 1 - (np.count_nonzero(Y_pred - Y_label) / Y_label.shape[0])
	print("Test accuracy: ", test_acc)
	return np.mean(scores), test_acc

    
val_score, test_score = [], []
reg = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
for r in reg:
	cross_score, testing_score = train_on_SVM(r)
	#print("score: %s, %s" % (cross_score, test_score))
	val_score.append(cross_score)
	test_score.append(testing_score)

print(val_score)
print(test_score)
plt.plot(reg, val_score, label="Cross-Validation Score")
plt.plot(reg, test_score, label="Test Score")
plot_attributes("Regularization Value", "Accuracy", "Regularization Parameter v/s Accuracy")










