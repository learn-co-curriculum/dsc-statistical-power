
# Statistical Power

## Introduction


You've started to investigate hypothesis testing, p-values and their use for accepting or rejecting the null hypothesis. With this, the power of a statistical test measures an experiment's ability to detect a difference, when one exists. In the case of testing whether a coin is fair, the power of our statistical test would be the probability of rejecting the null hypothesis "this coin is fair" when the coin was unfair. As you might assume, the power of this statistical test would thus depend on several factors including our p-value threshold for rejecting the null hypothesis, the size of our sample and the 'level of unfairness' of the coin in question.

## Objectives

You will be able to:

* Describe the concept of “Power” in relation to p-value and effect size for hypothesis testing
* Understand and critically evaluate the factors influencing the power of an experiment
* Perform Power calculation using SciPy and Python
* Demonstrate the impact of sample size on statistical power using simulations
* Demonstrate the combined effect of sample size and effect size on statistical power using simulations  

## The Power of a Statistical Test

The power of a statistical test is defined as the probability of rejecting the null hypothesis, given that it is indeed false. As with any probability, the power of a statistical test therefore ranges from 0 to 1, with 1 being a perfect test that gaurantees rejecting the null hypothesis when it is indeed false.  

Intrinsically, this is related to $\beta$, the probability of type II errors. When designing a statistical test, a researcher will typically determine an acceptable $\alpha$, such as .05, the probability of type I errors. (Recall that type I errors are when the null-hypothesis is rejected when actually true.) From this given alpha value, an optimal threshold for rejecting the null-hypothesis can be determined. That is, for a given $\alpha$ value, you can calculate a threshold which maximizes the power of the test. For any given $\alpha$, $power = 1 - \beta$.



> Note: Ideally, $\alpha$ and $\beta$ would both be minimized, but this is often costly, impracticle or impossible depending on the scenario and required sample sizes. 

## Effect Size

The effect size is the magnitude of the difference you are testing between the two groups. Thus far, you've mainly been investigating the mean of a sample. For example, after flipping a coin n number of times, you've investigated using a t-test to determine whether the coin is a fair coin (p(heads)=0.5). To do this, you compare the mean of the sample to that of another sample, if comparing coins, or to a know theoretical distribution. Similarly, you might compare the mean income of a sample population to that of a census tract to determine if the populations are statistically different. In such cases, Cohen's D is typically the metric used as the effect size. Cohen's d is defined as:  
$ d = \frac{m_1 - m_2}{s}$,  where m_1 and m_2 are the respective sample means and s is the (overall) standard deviation

That is the difference of the means divided by the standard deviation.  

## Power Analysis

Since $\alpha$, power, sample size and effect size are all related quantities, you can take a look at some plots of the power of some t-tests, given varying sample sizes. This will allow you to develop a deeper understanding of how these quantities are related and what constitutes a convincing statistical test.  

To start, imagine the scenario of detecting an unfair coin. With this, the null-hypothesis $H_0(heads) = 0.5$. From here, the power will depend on both the sample size and the effect size (that is the threshold for the alternative hypothesis). For example, if the alternative hypothesis has a large margin from the null-hypothesis such as $H_a(heads) = 0.8$, then the likelihood of rejecting the null-hypothesis is diminished; an unfair coin where $P(heads)=.6$ for example, is not apt to lead to the reject of the null-hypothesis in the current formulation.   

To start, you might choose an alpha value based that you are willing to accept such as $\alpha=0.05$. From there, you could observe the power of various statistical test against various sample and effect sizes.  

For example, if we wish to state the alternative hypothesis $H_a = .55$, then the effect size (using Cohen's d) would be:

$ d = \frac{m_1 - m_2}{s}$  
$ d = \frac{.55 - .5}{s}$

Furthermore, since we are dealing with a binomial variable, the standard deviation of the sample should follow the formula $\sqrt{n\bullet p(1-p)}$.  
So some potential effect size values for various scenarios might look like this:


```python
m1 = .55
m2 = .5
p = m2
rows = []
for n in [10, 20, 50, 500]:
    std = np.sqrt(n*p*(1-p))
    d = (m1-m2)/std
    rows.append({'Effect_Size': d, 'STD': std, 'Num_observations': n})
print('Hypothetical Effect sizes for p(heads)=.55 vs p(heads)=.5')
pd.DataFrame(rows)
```

    Hypothetical Effect sizes for p(heads)=.55 vs p(heads)=.5





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Effect_Size</th>
      <th>Num_observations</th>
      <th>STD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.031623</td>
      <td>10</td>
      <td>1.581139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.022361</td>
      <td>20</td>
      <td>2.236068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.014142</td>
      <td>50</td>
      <td>3.535534</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.004472</td>
      <td>500</td>
      <td>11.180340</td>
    </tr>
  </tbody>
</table>
</div>



As a general rule of thumb, all of these effect sizes are quite small. here's the same idea expanded to other alternative hypotheses:


```python
m2 = .5
rows = {}
for n in [10, 20, 50, 500]:
    temp_dict = {}
    for m1 in [.51, .55, .6, .65, .7, .75, .8, .85, .9]:
        p = m1
        std = np.sqrt(n*p*(1-p))
        d = (m1-m2)/std
        temp_dict[m1] = d
    rows[n] = temp_dict
print('Hypothetical Effect Sizes for Various Alternative Hypotheses')
df = pd.DataFrame.from_dict(rows, orient='index')
# df.index = [10,20,50, 500]
# df.index.name = 'Sample_Size'
# df.columns.name = 'Alternative Hypothesis'
df
```

    Hypothetical Effect Sizes for Various Alternative Hypotheses





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.51</th>
      <th>0.55</th>
      <th>0.6</th>
      <th>0.65</th>
      <th>0.7</th>
      <th>0.75</th>
      <th>0.8</th>
      <th>0.85</th>
      <th>0.9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.006326</td>
      <td>0.031782</td>
      <td>0.064550</td>
      <td>0.099449</td>
      <td>0.138013</td>
      <td>0.182574</td>
      <td>0.237171</td>
      <td>0.309965</td>
      <td>0.421637</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.004473</td>
      <td>0.022473</td>
      <td>0.045644</td>
      <td>0.070321</td>
      <td>0.097590</td>
      <td>0.129099</td>
      <td>0.167705</td>
      <td>0.219179</td>
      <td>0.298142</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.002829</td>
      <td>0.014213</td>
      <td>0.028868</td>
      <td>0.044475</td>
      <td>0.061721</td>
      <td>0.081650</td>
      <td>0.106066</td>
      <td>0.138621</td>
      <td>0.188562</td>
    </tr>
    <tr>
      <th>500</th>
      <td>0.000895</td>
      <td>0.004495</td>
      <td>0.009129</td>
      <td>0.014064</td>
      <td>0.019518</td>
      <td>0.025820</td>
      <td>0.033541</td>
      <td>0.043836</td>
      <td>0.059628</td>
    </tr>
  </tbody>
</table>
</div>



While a bit long winded, you can see that realalistic effect sizes for this scenario could be anywhere from 0.05 (or lower) up to approximately .4. With that, here's how you could use statsmodels to do a power analysis of various experimental designs.


```python
from statsmodels.stats.power import TTestIndPower, TTestPower
```


```python
power_analysis = TTestIndPower()
```


```python
power_analysis.plot_power(dep_var="nobs",
                          nobs = np.array(range(5,1500)),
                          effect_size=np.array([.05, .1, .2,.3,.4,.5]),
                          alpha=0.05)
plt.show()
```


![png](index_files/index_7_0.png)


As this should demonstrate, detecting small perturbances can be quite difficult!

## Additional Resources

* [Stats Models Documentation](http://www.statsmodels.org/dev/generated/statsmodels.stats.power.TTestIndPower.html)

## Summary

In this lesson, you learned about the idea of "statistical power" and how sample size, alpha and effect size impact the power of an experiment. Remember, the power of a statistical test is the probability of rejecting the null hypothesis when it is indeed false.
