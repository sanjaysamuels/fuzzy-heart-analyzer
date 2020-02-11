import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Generate universe variables (Input variables)
#   * HeartRate, BloodPressure and OutsideTemperature on subjective ranges [0, 10]
x_HeartRate = np.arange(0, 11, 1)
x_BloodPressure = np.arange(0, 11, 1)
x_OutsideTemperature = np.arange(0, 11, 1)

# Output Variable
#   * Body condition has a range of 0 - 100
x_BodyCondition  = np.arange(0, 101, 1)

# Below we develop the inputs of the fuzzy membership functions

'''
If HeartRate is low then the patient is in rest state
If HeartRate is medium then the patient is in a normal state
If HeartRate is high then the patient is in a active state which may be bad for the body
'''
HeartRate_lo = fuzz.trimf(x_HeartRate, [0, 0, 5])
HeartRate_md = fuzz.trimf(x_HeartRate, [0, 5, 10])
HeartRate_hi = fuzz.trimf(x_HeartRate, [5, 10, 10])

'''
If BloodPressure is low then the patient is in a bad state
If BloodPressure is medium then the patient is in a good state
If BloodPressure is good then the patient is in a bad state
'''
BloodPressure_lo = fuzz.trimf(x_BloodPressure, [0, 0, 5])
BloodPressure_md = fuzz.trimf(x_BloodPressure, [0, 5, 10])
BloodPressure_hi = fuzz.trimf(x_BloodPressure, [5, 10, 10])

'''
If OutsideTemperature is low then the patient is in a cold state which is not good for the heart
If OutsideTemperature is medium then the patient is in a good state
If OutsideTemperature is good then the patient is in a hot state which is not good for the heart
'''
OutsideTemperature_lo = fuzz.trimf(x_OutsideTemperature, [0, 0, 5])
OutsideTemperature_md = fuzz.trimf(x_OutsideTemperature, [0, 5, 10])
OutsideTemperature_hi = fuzz.trimf(x_OutsideTemperature, [5, 10, 10])


# Output of the fuzzy memebership function,
BodyCondition_lo = fuzz.trimf(x_BodyCondition, [0, 0, 50])
BodyCondition_md = fuzz.trimf(x_BodyCondition, [0, 50, 100])
BodyCondition_hi = fuzz.trimf(x_BodyCondition, [50, 100, 100])


# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

# Plotting the input functions and output function on a graph
ax0.plot(x_HeartRate, HeartRate_lo, 'b', linewidth=1.5, label='Poor')
ax0.plot(x_HeartRate, HeartRate_md, 'g', linewidth=1.5, label='Normal')
ax0.plot(x_HeartRate, HeartRate_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('Heart Rate')
ax0.legend()

ax1.plot(x_BloodPressure, BloodPressure_lo, 'b', linewidth=1.5, label='Low')
ax1.plot(x_BloodPressure, BloodPressure_md, 'g', linewidth=1.5, label='Medium')
ax1.plot(x_BloodPressure, BloodPressure_hi, 'r', linewidth=1.5, label='High')
ax1.set_title('Blood Pressure')
ax1.legend()

ax2.plot(x_OutsideTemperature, OutsideTemperature_lo, 'b', linewidth=1.5, label='Cold')
ax2.plot(x_OutsideTemperature, OutsideTemperature_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_OutsideTemperature, OutsideTemperature_hi, 'r', linewidth=1.5, label='Hot')
ax2.set_title('Outside Temperature')
ax2.legend()

ax3.plot(x_BodyCondition, BodyCondition_lo, 'b', linewidth=1.5, label='Bad')
ax3.plot(x_BodyCondition, BodyCondition_md, 'g', linewidth=1.5, label='Medium')
ax3.plot(x_BodyCondition, BodyCondition_hi, 'r', linewidth=1.5, label='Good')
ax3.set_title('Body Condition')
ax3.legend()

# To remove the top and right axis
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()



# We need the activation of our fuzzy membership functions at these values.
'''
The exact values of 2.5, 3 and 1 will be intrepreted in their respective graphs on how much
each of the value belong to their respective low, medium and high functions
'''
HeartRate_level_lo = fuzz.interp_membership(x_HeartRate, HeartRate_lo, 5)
HeartRate_level_md = fuzz.interp_membership(x_HeartRate, HeartRate_md, 5)
HeartRate_level_hi = fuzz.interp_membership(x_HeartRate, HeartRate_hi, 5)

BloodPressure_level_lo = fuzz.interp_membership(x_BloodPressure, BloodPressure_lo, 5)
BloodPressure_level_md = fuzz.interp_membership(x_BloodPressure, BloodPressure_md, 5)
BloodPressure_level_hi = fuzz.interp_membership(x_BloodPressure, BloodPressure_hi, 5)

OutsideTemperature_level_lo = fuzz.interp_membership(x_OutsideTemperature, OutsideTemperature_lo, 8)
OutsideTemperature_level_md = fuzz.interp_membership(x_OutsideTemperature, OutsideTemperature_md, 8)
OutsideTemperature_level_hi = fuzz.interp_membership(x_OutsideTemperature, OutsideTemperature_hi, 8)



'''
Now we begin the formation of the Rules

The rules to determine a patients condition are

Rule1:
If the heart rate is high and the blood pressure is low, or the outside temperature is low then the body condition of the patient is low (critical).
'''
active_rule1 = np.fmin(HeartRate_level_hi, BloodPressure_level_lo) # and funcion is used here
activate_rule2 = np.fmax(active_rule1, OutsideTemperature_level_lo) # or function is used here (where the maximum of these two are taken)
BodyCondition_activation_lo = np.fmin(activate_rule2, BodyCondition_lo) # membership function formed using fmin

'''
Rule2:
If the heart rate is low and the blood pressure is medium, or the outside temperature is medium then the patient’s body condition is stable.
'''
active_rule3 = np.fmin(HeartRate_level_lo, BloodPressure_level_md)
active_rule4 = np.fmax(active_rule3, OutsideTemperature_level_md)
BodyCondition_activation_md = np.fmin(active_rule4, BodyCondition_md)

'''
Rule3:
If the blood pressure is medium and the outside temperature is medium, and the heart rate is medium, then the patient’s body condition is good, and no change is necessary.
'''
active_rule5 = np.fmin(BloodPressure_level_md, OutsideTemperature_level_md)
active_rule6 = np.fmin(active_rule5, HeartRate_level_md)
BodyCondition_activation_hi = np.fmin(active_rule6, BodyCondition_hi)

BodyCondition0 = np.zeros_like(x_BodyCondition)



# Plotting the graph and visualize for the three rules made above
fig, ax0 = plt.subplots(figsize=(9, 4))

ax0.fill_between(x_BodyCondition, BodyCondition0, BodyCondition_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_BodyCondition, BodyCondition_lo, 'b', linewidth=0.5, linestyle='-', )
ax0.fill_between(x_BodyCondition, BodyCondition0, BodyCondition_activation_md, facecolor='r', alpha=0.7)
ax0.plot(x_BodyCondition, BodyCondition_md, 'r', linewidth=0.5, linestyle='-')
ax0.fill_between(x_BodyCondition, BodyCondition0, BodyCondition_activation_hi, facecolor='c', alpha=0.7)
ax0.plot(x_BodyCondition, BodyCondition_hi, 'c', linewidth=0.5, linestyle='-')

ax0.set_title('Output for the membership activity')



# To remove the top and right axis
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()



# Now we aggrigate the three inputs to for a crisp value

aggregated_1 = np.fmax(BodyCondition_activation_lo,
                     np.fmax(BodyCondition_activation_md, BodyCondition_activation_hi))



# Now we calculate the deffuzified result for Agregated 1
BodyCondition = fuzz.defuzz(x_BodyCondition, aggregated_1, 'centroid')

BodyCondition_activation = fuzz.interp_membership(x_BodyCondition, aggregated_1, BodyCondition)  # for plotting the graph
print("BC", BodyCondition)
print("BA", BodyCondition_activation)

if BodyCondition_activation < 0.5:
    print("The patient's body condtion is critical")

elif BodyCondition_activation >= 0.5 and BodyCondition <= 0.8:
    print("The patient's body condtion is stable")

elif BodyCondition_activation > 0.8:
    print("The patient's body condtion is healthy")


# Visualize the plot for aggregated 1
fig, ax0 = plt.subplots(figsize=(9, 4))

ax0.plot(x_BodyCondition, BodyCondition_lo, 'b', linewidth=0.5, linestyle='-', )
ax0.plot(x_BodyCondition, BodyCondition_md, 'g', linewidth=0.5, linestyle='-')
ax0.plot(x_BodyCondition, BodyCondition_hi, 'r', linewidth=0.5, linestyle='-')
ax0.fill_between(x_BodyCondition, BodyCondition0, aggregated_1, facecolor='Orange', alpha=0.7)
ax0.plot([BodyCondition, BodyCondition], [0, BodyCondition_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated 1 degree of membership and result (line)')


# To remove the top and right axis
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
