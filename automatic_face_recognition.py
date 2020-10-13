from __future__ import print_function
import face_recognition
import copy
import numpy as np
import matplotlib.pylab as plt
import math
from itertools import combinations
from scipy.stats.stats import pearsonr
plt.style.use("../misc/paper_style.mplstyle")
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score

# Load target images
target_image_bg1 = face_recognition.load_image_file("imagesBW/target_bg1.pgm")
target_image_bg2 = face_recognition.load_image_file("imagesBW/target_bg2.pgm")

# Get the face encodings for each face in each image file
target_face_encoding_bg1 = face_recognition.face_encodings(target_image_bg1)[0]
target_face_encoding_bg2 = face_recognition.face_encodings(target_image_bg2)[0]

# Get the stimuli used in the experiment
stimuli = np.loadtxt("results/Subj_001_stimuli.txt", skiprows=1, delimiter=",", dtype=bytes, usecols=(1),
                     converters={1: lambda s: s[3:]}).astype(str)
# Reorder stimuli to sessions 0, 1, 2, 3, 4, 5
session_order = list(map(int, open("results/Subj_001_stimuli.txt", "r").readline().rstrip().split(" ")))
trials = int(open("results/Subj_001_errors.txt", "r").readline().rstrip().split(" ")[1])
reorder_indices = []
for sess in session_order:
    reorder_indices += list(range(sess * trials, (sess + 1) * trials))
tmp = np.asarray(stimuli)
tmp[reorder_indices] = copy.deepcopy(tmp)
stimuli = tmp

tp = {"L": 0, "C": 0, "R": 0}
tn = {"L": 0, "C": 0, "R": 0}
fp = {"L": 0, "C": 0, "R": 0}
fn = {"L": 0, "C": 0, "R": 0}
num_bg1 = 0
num_bg2 = 0

# Save results in a file
nn_correctness = np.zeros(shape=(len(stimuli), ))
nn_confidence = np.zeros(shape=(len(stimuli), ))

kf = KFold(n_splits=8, shuffle=False, random_state=0)
scores = np.zeros(shape=(len(stimuli), ))
labels = np.zeros(shape=(len(stimuli), ))

print("Loading and encoding stimuli")
stim_encodings = []
for trial, stim in enumerate(stimuli):
    img = face_recognition.load_image_file("imagesBW/%s.jpg" % stim)
    stim_encodings.append(face_recognition.face_encodings(img))
    labels[trial] = 1 if "T" in stim else 0
print("Running CV")

for train_index, test_index in kf.split(stimuli):
    # Find the best threshold, which is the one that maximises accuracy on the training set
    possible_thresholds = np.arange(0.5, 1.0, 0.0005)
    fitness_per_threshold = np.zeros((len(possible_thresholds), 1))
    for threshold in possible_thresholds:
        decisions_per_fold = []
        confidence_per_fold = []
        for trial in train_index:
            if "bg1" in stimuli[trial]:
                target_face_encoding = target_face_encoding_bg1
            elif "bg2" in stimuli[trial]:
                target_face_encoding = target_face_encoding_bg2
            else:
                print("Background not recognised, skip stimulus")
                continue
            results = face_recognition.compare_faces(stim_encodings[trial], target_face_encoding, tolerance=threshold)
            decisions_per_fold.append(np.any(results))
            confidence = threshold
            if stim_encodings[trial]:
                confidence = min(face_recognition.face_distance(stim_encodings[trial], target_face_encoding))
                if confidence < threshold:
                    confidence = 1 - confidence / threshold
                else:
                    confidence = (confidence - threshold) / (1 - threshold)
            confidence_per_fold.append(confidence)
        #fitness_per_threshold[list(possible_thresholds).index(threshold)] = f1_score(labels[train_index], decisions_per_fold)
        fitness_per_threshold[list(possible_thresholds).index(threshold), 0] = accuracy_score(labels[train_index], decisions_per_fold)
    best_threshold = possible_thresholds[np.argmax(fitness_per_threshold, axis=0)[0]]
    print("FOLD best threshold", best_threshold)
    # Testing
    for trial in test_index:
        print(stimuli[trial], end=" ")
        if "bg1" in stimuli[trial]:
            target_face_encoding = target_face_encoding_bg1
            num_bg1 += 1
        elif "bg2" in stimuli[trial]:
            target_face_encoding = target_face_encoding_bg2
            num_bg2 += 1
        else:
            print("Background not recognised, skip stimulus")
            continue
        confidence = 0
        if stim_encodings[trial]:
            confidence = min(face_recognition.face_distance(stim_encodings[trial], target_face_encoding))
            scores[trial] = confidence
            if confidence < best_threshold:
                confidence = 1 - confidence / best_threshold
            else:
                confidence = (confidence - best_threshold) / (1 - best_threshold)
        else:
            scores[trial] = best_threshold
        # Squashing function (sigmoid)
        nn_confidence[trial] = 2/(1+np.exp(-10*confidence))-1
        results = face_recognition.compare_faces(stim_encodings[trial], target_face_encoding, tolerance=best_threshold)
        if labels[trial] and np.any(results):
            # Target correct
            print("TARGET", end=" ")
            nn_correctness[trial] = -1
            if "L" in stimuli[trial]:
                tp["L"] += 1
            elif "C" in stimuli[trial]:
                tp["C"] += 1
            elif "R" in stimuli[trial]:
                tp["R"] += 1
        elif labels[trial] and not np.any(results):
            # Target incorrect
            nn_correctness[trial] = 1
            if "L" in stimuli[trial]:
                fn["L"] += 1
            elif "C" in stimuli[trial]:
                fn["C"] += 1
            elif "R" in stimuli[trial]:
                fn["R"] += 1
        elif not labels[trial] and not np.any(results):
            # Nontarget correct
            nn_correctness[trial] = -1
            if "L" in stimuli[trial]:
                tn["L"] += 1
            elif "C" in stimuli[trial]:
                tn["C"] += 1
            elif "R" in stimuli[trial]:
                tn["R"] += 1
        elif not labels[trial] and np.any(results):
            # Nontarget incorrect
            print("TARGET", end=" ")
            nn_correctness[trial] = 1
            if "L" in stimuli[trial]:
                fp["L"] += 1
            elif "C" in stimuli[trial]:
                fp["C"] += 1
            elif "R" in stimuli[trial]:
                fp["R"] += 1
        else:
            print("This is impossible")
            exit(1)
        print(str(nn_confidence[trial]))

# Plot overall ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color="k")
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.yticks(np.arange(0, 1.01, 0.1), np.arange(0, 1.01, 0.1))
plt.xticks(np.arange(0, 1.01, 0.1), np.arange(0, 1.01, 0.1))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

# create the axis of thresholds (scores)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, linestyle=':', color="b", markeredgecolor="b")
ax2.set_ylabel('Threshold',color='b')
ax2.set_ylim([0.0, 1.01])
#ax2.set_ylim([thresholds[-1], thresholds[0]])
ax2.set_yticks(np.arange(0, 1.01, 0.1))
ax2.set_yticklabels(np.arange(0, 1.01, 0.1))
ax2.set_xlim([0, 1.01])

plt.savefig("analysis/automatic_face_recognition_roc.pdf")

np.savetxt("results/neural_network_decisions.txt", nn_correctness, fmt="%d", delimiter="\t")
np.savetxt("results/neural_network_confidence.txt", nn_confidence, fmt="%.8f", delimiter="\t")
np.savetxt("results/neural_network_scores.txt", scores, fmt="%.8f", delimiter="\t")

print()
print("Accuracy: %.2f %%" % ((tp["L"]+tp["C"]+tp["R"]+tn["L"]+tn["C"]+tn["R"]) / (num_bg1+num_bg2) * 100))
print("> TP:", tp["L"]+tp["C"]+tp["R"])
print("> TN:", tn["L"]+tn["C"]+tn["R"])
print("> FP:", fp["L"]+fp["C"]+fp["R"])
print("> FN:", fn["L"]+fn["C"]+fn["R"])
print("> Total stimuli:", num_bg1+num_bg2)
print()
print("Accuracy LEFT: %.2f %%" % (3 * (tp["L"]+tn["L"]) / (num_bg1+num_bg2) * 100))
print("> TP:", tp["L"])
print("> TN:", tn["L"])
print("> FP:", fp["L"])
print("> FN:", fn["L"])
print("Accuracy CENTRE: %.2f %%" % (3 * (tp["C"]+tn["C"]) / (num_bg1+num_bg2) * 100))
print("> TP:", tp["C"])
print("> TN:", tn["C"])
print("> FP:", fp["C"])
print("> FN:", fn["C"])
print("Accuracy RIGHT: %.2f %%" % (3 * (tp["R"]+tn["R"]) / (num_bg1+num_bg2) * 100))
print("> TP:", tp["R"])
print("> TN:", tn["R"])
print("> FP:", fp["R"])
print("> FN:", fn["R"])


# If you want to load the results from file instead of rerunning the ResNet, comment the lines above
# and uncomment the following ones
"""
nn_decisions = np.loadtxt("results/neural_network_decisions.txt", delimiter="\t")
nn_confidence = np.loadtxt("results/neural_network_confidence.txt", delimiter="\t")
# Reorder stimuli to sessions 0, 1, 2, 3, 4, 5
stimuli = np.loadtxt("results/Subj_001_stimuli.txt", skiprows=1, delimiter=",", dtype=bytes, usecols=(1),
                     converters={1: lambda s: s[3:]}).astype(str)
session_order = list(map(int, open("results/Subj_001_stimuli.txt", "r").readline().rstrip().split(" ")))
trials = int(open("results/Subj_001_errors.txt", "r").readline().rstrip().split(" ")[1])
reorder_indices = []
for sess in session_order:
    reorder_indices += list(range(sess * trials, (sess + 1) * trials))
tmp = np.asarray(stimuli)
tmp[reorder_indices] = copy.deepcopy(tmp)
stimuli = tmp
#print(stimuli)
"""
mask_left = np.array([True] * trials + [False] * trials * 2 + [True] * trials + [False] * trials * 2)
mask_central = np.array([False] * trials + [True] * trials + [False] * trials +
                        [False] * trials + [True] * trials + [False] * trials)
mask_right = np.array([False] * trials * 2 + [True] * trials + [False] * trials * 2 + [True] * trials)
majority_decisions = np.zeros((trials * 2))
correctness_per_image = np.zeros((trials * 2, 3))
confidence_per_image = np.zeros((trials * 2, 3))
stim_list = stimuli.tolist()

for trial_index, stim_left in enumerate(stimuli[mask_left]):
    append = "_T" if "_T" in stim_left else ""
    stimulus_key = stim_left[:(7 if "bg1" in stim_left else 8)]
    correctness_per_image[trial_index, 0] = nn_correctness[stim_list.index(stimulus_key+"L"+append)]
    confidence_per_image[trial_index, 0] = nn_confidence[stim_list.index(stimulus_key + "L" + append)]
    correctness_per_image[trial_index, 1] = nn_correctness[stim_list.index(stimulus_key+"C"+append)]
    confidence_per_image[trial_index, 1] = nn_confidence[stim_list.index(stimulus_key + "C" + append)]
    correctness_per_image[trial_index, 2] = nn_correctness[stim_list.index(stimulus_key+"R"+append)]
    confidence_per_image[trial_index, 2] = nn_confidence[stim_list.index(stimulus_key + "R" + append)]

# Compute majority decision
majority_decisions = np.asarray([1 if sum(decisions_image) > 0 else -1 for decisions_image in correctness_per_image])
print("> MAJORITY: %.2f %%" % (np.sum(majority_decisions == -1) / correctness_per_image.shape[0] * 100))
tp = 0
tn = 0
fp = 0
fn = 0
for i, decision in enumerate(majority_decisions):
    if "_T" in stimuli[i]:
        if decision == 1:
            fn += 1
        elif decision == -1:
            tp += 1
    else:
        if decision == 1:
            fp += 1
        elif decision == -1:
            tn += 1
print("> TP:", tp)
print("> TN:", tn)
print("> FP:", fp)
print("> FN:", fn)

# Compute confidence-based decisions
confidence_majority_decisions = np.asarray([1 if sum(correctness_per_image[trial]*confidence_per_image[trial]) > 0 else -1
                                            for trial in range(correctness_per_image.shape[0])])

print("> CONFIDENCE MAJORITY: %.2f %%" % (np.sum(confidence_majority_decisions == -1) /
                                          correctness_per_image.shape[0] * 100))
tp = 0
tn = 0
fp = 0
fn = 0
for i, decision in enumerate(confidence_majority_decisions):
    if "_T" in stimuli[i]:
        if decision == 1:
            fn += 1
        elif decision == -1:
            tp += 1
    else:
        if decision == 1:
            fp += 1
        elif decision == -1:
            tn += 1
print("> TP:", tp)
print("> TN:", tn)
print("> FP:", fp)
print("> FN:", fn)

print("Pearson's correlation and p-values between decisions")
for vp_pair in combinations(range(correctness_per_image.shape[-1]), 2):
    cc, p = pearsonr(correctness_per_image[:, vp_pair[0]],
                     correctness_per_image[:, vp_pair[1]])
    print(vp_pair, "%.3f %.3f" % (cc, p))
