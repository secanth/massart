<h1>Code for "Classification Under Misspecification: Halfspaces, Generalized Linear Models, and Connections to Evolvability"</h1>

These are scripts for reproducing our paper "Classification Under Misspecification: Halfspaces, Generalized Linear Models, and Connections to Evolvability" (<a href="https://arxiv.org/abs/2006.04787">preprint</a>).

<h2>Contents</h2>


<ul>
  <li><code>adult.csv</code> contains a copy of the <a href="http://archive.ics.uci.edu/ml/datasets/Adult">UCI Adult dataset</a>.
  <li><code>gauss/</code> contains the results of our synthetic experiment</li>
  <li><code>real/</code> contains the results of our UCI Adult experiment</li>
  <ul>
    <li><code>noncensor/</code> contains the experiment where demographic fields were not hidden</li>
    <li><code>censor/</code> contains the experiment where demographic fields were hidden</li>
  </ul>
  <li>To reproduce the plots in this work, run <code>python plots.py</code></li>
  <li><code>experiment.py</code> is the wrapper for running both our real and synthetic experiments. Setting the flag <code>EXPERIMENT_TYPE</code> therein to one of ['real', 'synthetic-rcn', 'synthetic-massart'] and running <code>python experiment.py</code> will populate <code>gauss/</code> and <code>real/</code> with the relevant experimental data (these folders have been pre-populated for the user's convenience). </li>
</ul>
