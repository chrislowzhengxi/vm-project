# Speaker notes: Optimizing Cloud VM CPU Reservations (8 minutes)

The main deck is 9 slides, followed by 4 backup slides. The same notes are embedded in the PowerPoint file (Notes pane).

* **Spoken length:** about 955 words including transitions, roughly 7:40 at a relaxed 125 words per minute. That
  leaves about 20 seconds of slack for pauses at the charts.
* **Speakers:** Chris presents slides 1–5 (4:20) and Emma slides 6–9 (3:40). Swap freely.

## Rehearsal timing

| # | Slide | Speaker | Target | Running total | Spoken words |
|---|---|---|---:|---:|---:|
| 1 | Title and question | Chris | 0:20 | 0:20 | ~45 |
| 2 | The reservation decision | Chris | 0:55 | 1:15 | ~115 |
| 3 | Data and chronological split | Chris | 0:45 | 2:00 | ~95 |
| 4 | Methods and the exact OPT tree | Chris | 1:10 | 3:10 | ~140 |
| 5 | Main results at β = 5 | Chris → Emma | 1:10 | 4:20 | ~125 |
| 6 | Customer history | Emma | 0:55 | 5:15 | ~110 |
| 7 | Prediction vs prescription | Emma | 0:55 | 6:10 | ~110 |
| 8 | Simplicity and risk | Emma | 1:10 | 7:20 | ~110 |
| 9 | Conclusions and limitation | Emma | 0:40 | 8:00 | ~90 |

**Checkpoints for rehearsal:** be on slide 4 by 2:00, on slide 6 by 4:20, and on slide 9 by 7:20. If you are behind at
slide 6:
* **Slide 6:** drop the sentence about greedy and the forest.
* **Slide 8:** drop the exceedance-number sentence.

---

## Slide 1: Title (Chris, 0:20)

**Main message:** Customers ask for more CPU than they use. When a VM is created, how much should the provider set aside?

**Script:** Hi, I'm Chris, and this is Emma. When customers create a virtual machine, they ask for a number of CPUs, but most VMs use much less. So our question is simple: when a VM is created, how much CPU should the provider actually reserve?

**Transition:** Here is what that decision looks like for one VM.

## Slide 2: The reservation decision (Chris, 0:55)

**Main message:** We pick one of four reservation levels, and a simple cost trades wasted capacity against shortfall.

**Script:** Take a VM that requests 8 virtual CPUs. Its 95th-percentile usage turns out to be 40 percent of that request. We can reserve 25, 50, 75 or 100 percent. Our cost charges for what we reserve, plus beta times any demand above the reservation. With beta equal to 5, being short costs five times as much as reserving. Reserving 25 percent leaves the VM 15 points short, so it costs 1.0. Reserving 50 percent costs 0.5, the best choice here. Reserving 100 percent is safe, but it also costs 1.0. To be clear, this is a modeled cost. We score hypothetical reservations on real traces. We never resized any Azure VMs.

**Transition:** So where do those traces come from?

## Slide 3: Data and design (Chris, 0:45)

**Main message:** Real Azure data, features known at creation time, and a time-based split with no leakage.

**Script:** We use the Azure Public Dataset: about 2.5 million VMs created over 30 days in one region. Each VM has 26 features known at creation time, including its size and the history of the same customer and deployment. We split by time. We train on the first 21 days, tune on the next four, and test on the last five. Two rules prevent leakage. Customer history only uses VMs that had already finished. And a training VM only counts if it ended before the cutoff.

**Transition:** With that data, we compared four ways to make the decision.

## Slide 4: Methods and the exact OPT tree (Chris, 1:10)

**Main message:** Exact OPT turns customer history into a short, readable rule, and our solver certifies it is the best tree of its kind on the training data.

**Script:** The first approach predicts P95 usage and rounds up to the next tier. The other three learn from the cost itself. A greedy policy tree picks one split at a time. A cost-aware random forest estimates the shortfall at each tier and picks the cheapest. Our focus is the exact optimal policy tree. This is our actual tree for beta equals 5. It first asks how often the customer's past VMs ran above 75 percent. Near-idle customers, about 16 percent of VMs, get 25 percent. Customers with frequent hot VMs mostly get 100. We wrote our own dynamic program to find it. It checks every combination of splits on 8 features with 31 thresholds each. So within that class, it has the certified lowest training cost, and it solves in about 5 seconds.

**Transition:** So how much does each approach save?

## Slide 5: Main results at β = 5 (Chris, 1:10)

**Main message:** Every cost-aware policy saves about 20 percent; exact OPT barely beats greedy, and the cost-aware forest is best but not interpretable.

**Script:** This chart shows savings in modeled cost against the best fixed rule. At beta equals 5, that rule reserves 100 percent for everyone. Predicting usage and rounding up saves 14 percent. Once a method accounts for the cost, savings jump to about 20 percent. The customer-history rule, the forest with a tuned margin, greedy and exact OPT all land within about a point of each other. Exact OPT saves 20.1 percent and greedy 20.0, and that difference is not statistically significant. OPT does reach a lower training cost, but that small edge doesn't carry over to new data. The cost-aware forest is best at 21.6 percent. But it's 100 trees, so nobody can read it.

**Handoff:** Emma will show where these savings come from.

## Slide 6: Customer history (Emma, 0:55)

**Main message:** Most of the value comes from the customer's own history, not from a fancier algorithm.

**Script:** Thanks. To see where the value comes from, we retrained each method on three feature sets. With only VM size and creation time, OPT saves 6 percent. Add the customer's history and it jumps to 20.3 percent. Adding deployment history barely changes anything. Greedy and the random forest show the same jump. Why does history matter so much? Almost every test VM comes from a returning customer. Their past VMs are basically a sample of how their next VM will behave, especially how often it runs hot. Once that is in, the choice of algorithm adds at most about one and a half points.

**Transition:** History tells us what to expect. The next question is how to turn that into a decision.

## Slide 7: Prediction vs prescription (Emma, 0:55)

**Main message:** With the same predictive model, choosing the reservation by expected cost beats rounding up a point prediction.

**Script:** Here we hold the prediction model fixed. All three rules use the same random forest. Rule A predicts P95 and rounds up, saving 14.2 percent. Rule B adds a safety margin tuned on validation and saves 19.8. Rule C uses the forest to estimate the whole distribution of demand, then picks the tier with the lowest expected cost. It saves 21.4 percent. Why? VMs predicted at 40 to 50 percent actually span all four tiers. Rounding up gives them all 50 percent, and 41 percent run over. Rule C adapts to each VM, and only 16 percent run over. So better predictions don't automatically give better decisions.

**Transition:** How complex does the policy itself need to be?

## Slide 8: Simplicity and risk (Emma, 1:10)

**Main message:** A much simpler tree keeps nearly all the savings, and the cheapest policy is not the one that exceeds its reservation least.

**Script:** A four-leaf tree, just two questions about customer history, saves 19.8 percent. That's 98.6 percent of the eight-leaf tree's savings, and the difference isn't statistically significant. That doesn't prove they're equivalent, but the extra leaves buy very little. The orange row is how often realized P95 lands above the reservation. It's not measured throttling. Notice the cheapest policy isn't the one that exceeds least: 7.7 percent for four leaves, 9.3 for eight, and 9.1 for the forest. Averages also hide differences between customers. The median customer saves 10.4 percent under OPT, about half the overall 20.1.

**Transition:** Let me pull this together.

## Slide 9: Conclusions (Emma, 0:40)

**Main message:** Customer history and the right objective deliver about 20 percent modeled savings; exact OPT adds readability and a certificate, not much extra accuracy.

**Script:** So, three takeaways. Customer history carries most of the signal. Deciding on cost beats rounding up predictions, even with the same model. And exact OPT gives a readable rule with a certified training optimum, but on test it only ties greedy, and the forest is about one and a half points better. One limitation: these are modeled savings on observed traces, not real billing or performance under enforced reservations. Overall, about 20 percent modeled savings, driven by customer history and the right objective. Thanks, we're happy to take questions.

---

## Likely questions and short answers

**What does β represent?**
It is the assumed cost of one unit of P95 demand above the reservation, relative to one unit of reserved capacity. It
is a modeling choice, not an Azure price or a measured SLA penalty.
* **Sensitivity:** we re-tuned every method at 12 values from 1.5 to 20 (backup slide 12).
* **Robustness:** the β = 5 tree costs less than 1% more than a re-tuned tree when the true β is between 4 and 8.

**Why is OPT "exact" if the Random Forest does better?**
"Exact" means our dynamic program finds the tree with the lowest training cost within a restricted class: depth ≤ 3,
8 features, 31 thresholds each, a minimum leaf size. The forest is a different, much more flexible class (100 deep
trees on 26 features). Restricted to the tree's 8 features, the forest still costs 0.787 against OPT's 0.799, so most
of its edge is flexibility, not information.

**Why does exact OPT barely beat greedy?**
Exact optimization lowers training cost by only 0.4–0.5% relative to greedy, and that is within the drift between the
training and test periods. One very informative variable family, the customer's past P95 distribution, dominates the
splits, so greedy rarely makes a bad first choice. On test the difference is 0.1 point at β = 5 and not significant.
Exactness does give stability: the exact tree is identical across random 200,000-VM training samples.

**Why are the customer-history features so important?**
* **Repeat customers:** 99.9% of test VMs belong to subscriptions with at least one completed earlier VM, and 91%
  to subscriptions with at least 1,000.
* **A direct sample:** a customer's past VMs are a sample of how its workloads behave, including how often they run
  hot.
* **The numbers:** OPT saves 6.0% with size and time features alone and 20.3% with customer history.

**Does lower modeled cost mean less physical CPU used?**
No. We compute a modeled reservation (OPT averages 73.4% of the request against 100%) and score it on observed
traces. We do not measure packing, consolidation, throttling or application performance under enforced reservations,
and we did not change any VM.

**Why P95 rather than average CPU?**
A reservation has to cover sustained high usage. The average understates the peaks a VM needs, while the maximum is a
single, noisy reading. P95 is a standard compromise. One caveat: for very short VMs (69% of test VMs run under an hour)
the P95 rests on few five-minute readings.

**How can the forest-weighted policy beat point prediction with the same forest?**
* **The cost depends on the tail:** it is asymmetric, so the best tier depends on the expected shortfall
  E[(Y − r)+ | x], not just the predicted mean.
* **Rule C uses the whole distribution:** it weights past VMs that share the forest's leaves with the new VM, which
  approximates the conditional distribution of demand (Bertsimas and Kallus).
* **Example:** VMs predicted at 40–50% actually span all four tiers. Rounding up leaves 41% of them exceeded;
  rule C leaves 16%.

**Why do smaller customers benefit less?**
The objective weights every VM equally, so large customers dominate the training cost and the tree's leaves are tuned
for them. Under OPT:
* **Median subscription:** with at least 5 test VMs, it saves 10.4%, against 20.1% overall.
* **Worse off:** 11.9% of subscriptions pay more than under always-100%, against 7.6–7.7% under greedy, the four-leaf
  tree or the forest.

This analysis is exploratory, not a fairness objective.

**Is the oracle achievable?**
No. It picks each VM's best tier using that VM's realized usage, which is unknown at creation time. Its 41.1% saving
is only a lower bound on cost.

**Did you implement the Homework 6 refinements?**
Yes. Both are in Section 5 of the report:
* **Forest-weighted prescription:** slide 7.
* **Interpretable clustering:** backup slide 13. k-means with K = 8, read through an Optimal Classification Tree,
  saves 19.7% at β = 5 against 20.1% for OPT; the difference is not significant. It falls behind at β = 10.
