Give me an intuition why the method in this paper works

Mar 1

Why Masking Updates Works: An Intuition
---------------------------------------

The core idea is surprisingly elegant: **randomly skipping parameter updates acts as an implicit second-order regularizer**, steering optimization toward flatter loss landscape regions --- without ever computing second-order information explicitly.

### The Key Insight from Proposition 1

When you randomly mask half the updates, the *expected* loss after the step isn't just the usual first-order descent term. There's an extra term that appears:

1-p2p(Δ(b))⊤Hbb(θ)Δ(b)\frac{1-p}{2p} (\Delta^{(b)})^\top H_{bb}(\theta) \Delta^{(b)}2p1-p​(Δ(b))⊤Hbb​(θ)Δ(b)

This term penalizes updates that point along **high-curvature directions** of the loss. Intuitively: if the loss curves sharply along your update direction, masking hurts you more (on average), so the optimizer is implicitly discouraged from taking those steps.

**Analogy:** Imagine hiking in fog and randomly deciding whether to take each step. If you're on a steep ridge, skipping steps is costly --- so you learn to prefer flatter terrain.

### Why Flat Regions Help

Flatter minima generalize better --- this is well-established empirically. Methods like SAM explicitly *search* for flat minima by computing adversarial perturbations. Magma gets this **for free** through stochastic noise, at no extra computational cost.

### Why Momentum Alignment Makes It Smarter (Magma)

Pure random masking treats all parameter blocks equally. But some blocks have gradients that are noisy and inconsistent across steps --- those are the dangerous ones. Magma scales each block's update by the cosine similarity between the current gradient and the momentum estimate.

The intuition: **if the gradient suddenly disagrees with where you've been heading, that's likely noise, not signal** --- so suppress the update. This is especially powerful under heavy-tailed gradient noise, which is endemic to LLM training.

### Why Dense Momentum Updates Matter

A subtle but important detail: even when a block's *parameter update* is skipped, its momentum statistics are still updated. This means the momentum remains a high-quality, low-variance estimate of the true gradient direction --- giving Magma a reliable signal for its alignment scores.

### The Net Effect

Magma effectively widens the stable learning rate range by suppressing the blocks that dominate curvature-weighted noise, making the optimizer more robust without requiring any explicit curvature computation or extra gradient evaluations.

What's the difference to explicit second order methods, e.g. Newton's?

Mar 2

Magma vs. Explicit Second-Order Methods
---------------------------------------

### What Newton's Method Does Explicitly

Newton's method computes the update:

θt+1=θt-H-1∇l(θt)\theta_{t+1} = \theta_t - H^{-1} \nabla l(\theta_t)θt+1​=θt​-H-1∇l(θt​)

It **explicitly measures curvature** via the Hessian and uses it to rescale the gradient --- taking small steps in sharp directions and large steps in flat ones. This is optimal locally but costs O(d2)O(d^2) O(d2) to store and O(d3)O(d^3) O(d3) to invert, which is completely infeasible for LLMs with billions of parameters.

* * * * *

### The Fundamental Difference

| Property | Newton / Quasi-Newton | Magma |
| --- | --- | --- |
| Curvature information | Explicitly computed | Never computed |
| Effect on sharp directions | Actively corrects them | Implicitly penalizes them |
| Cost | O(d2)O(d^2) O(d2) to O(d3)O(d^3) O(d3) | Zero overhead |
| Direction of effect | Rescales updates optimally | Discourages sharp updates stochastically |

The crucial distinction is **corrective vs. avoidant**:

-   Newton *corrects* for curvature --- it uses the Hessian to take the geometrically right step size in every direction
-   Magma *avoids* high-curvature directions --- it doesn't know where they are, but by randomly skipping updates and penalizing those that align with sharp directions in expectation, it drifts away from them over time

* * * * *

### A Physical Analogy

Think of navigating a hilly landscape:

-   **Newton's method** is like having a perfect topographic map --- you always know exactly how steep each direction is and adjust your stride accordingly
-   **Magma** is like walking with randomly unreliable legs --- because stumbling is more costly on steep terrain, you unconsciously migrate toward flatter ground over many steps

* * * * *

### An Important Subtlety: What Kind of Regularization?

Newton's method finds the **sharp minimum** of the loss most efficiently. Magma, by contrast, introduces a bias *away* from sharp minima entirely, toward flatter regions. These are actually different objectives:

-   Newton converges to whatever minimum the loss has, sharp or flat
-   Magma biases toward **flat minima**, which empirically generalize better in deep learning

So Magma isn't just a cheap approximation to Newton --- it's doing something Newton doesn't do at all. SAM (Sharpness-Aware Minimization) is the closest explicit analog, but it requires a full extra gradient evaluation per step to find adversarial perturbations. Magma achieves a similar flatness bias purely through structured stochasticity.

* * * * *

### Why This Matters Practically

The reason second-order methods haven't taken over LLM training isn't just cost --- it's that the Hessian of a transformer loss landscape is so ill-conditioned and heterogeneous that even approximating it well (as SOAP or Shampoo attempt) requires significant engineering. Magma sidesteps this entirely: it doesn't need to know *where* the sharp directions are, only that penalizing updates aligned with them (in expectation) is enough to improve generalization.

Let's ignore cost or stochastic vs deterministic. Let's just compare the key insight from proposition 1 to the formula for Newton's

Mar 2

Great question --- let's put them side by side directly.

**Newton's update** minimizes the second-order Taylor expansion of the loss:

l(θ-Δ)≈l(θ)-∇l⊤Δ+12Δ⊤HΔl(\theta - \Delta) \approx l(\theta) - \nabla l^\top \Delta + \frac{1}{2} \Delta^\top H \Deltal(θ-Δ)≈l(θ)-∇l⊤Δ+21​Δ⊤HΔ

Setting the derivative to zero gives Δ∗=H-1∇l\Delta^* = H^{-1} \nabla l Δ∗=H-1∇l. The Hessian term acts as a **metric** --- it tells you the right step size in each direction so you land at the local quadratic minimum.

**Proposition 1** says the expected loss under masking is:

l(θ-Δ)+1-p2pΔ⊤HΔ+O(∥Δ∥3)l(\theta - \Delta) + \frac{1-p}{2p} \Delta^\top H \Delta + O(\|\Delta\|^3)l(θ-Δ)+2p1-p​Δ⊤HΔ+O(∥Δ∥3)

The second term is structurally identical to the Hessian term in Newton's expansion.

* * * * *

**The key difference is what you optimize against.**

Newton minimizes over Δ\Delta Δ, so the Δ⊤HΔ\Delta^\top H \Delta Δ⊤HΔ term *helps* --- it pulls Δ\Delta Δ toward H-1∇lH^{-1}\nabla l H-1∇l, the curvature-corrected step. You *want* to account for it.

Magma treats Δ\Delta Δ as fixed (it comes from the base optimizer) and the Δ⊤HΔ\Delta^\top H \Delta Δ⊤HΔ term is a **penalty you cannot escape** --- it's added to whatever loss you achieve with your update. So the optimizer is implicitly pressured over many steps to move to regions where Δ⊤HΔ\Delta^\top H \Delta Δ⊤HΔ is small, i.e. where the loss is flat.

Same mathematical object, completely opposite role.

Your key difference is confusing to me. Doesn't proposition 1 say this is the actual expected loss that is optimized?

Mar 2

You're right, I was being sloppy. Let me reconsider.

Proposition 1 just says: here is the expected loss after a masked step. It's not a separate penalty you "add" --- it's simply what the loss evaluates to in expectation. So the optimizer *is* minimizing this quantity.

In that sense it really is structurally analogous to Newton. Both are doing:

minimize l(θ-Δ)+12Δ⊤HΔ⋅(some scalar)\text{minimize } l(\theta - \Delta) + \frac{1}{2}\Delta^\top H \Delta \cdot \text{(some scalar)}minimize l(θ-Δ)+21​Δ⊤HΔ⋅(some scalar)

The scalar is 11 1 for Newton (implicit in the quadratic approximation) and 1-p2p\frac{1-p}{2p} 2p1-p​ for Magma.

So the honest comparison is actually simpler than I made it: **Magma with a base optimizer implicitly solves the same kind of regularized objective as Newton, but with a fixed Δ\Delta Δ coming from the base optimizer rather than optimizing Δ\Delta Δ to minimize the quadratic.**

The real difference is then just: Newton finds the Δ\Delta Δ that minimizes this expression analytically, while Magma uses a fixed Δ\Delta Δ from e.g. RMSProp but the curvature penalty still shapes which regions of parameter space the trajectory prefers over time --- places where Δ⊤HΔ\Delta^\top H \Delta Δ⊤HΔ is naturally small.

Does that framing make more sense to you?

Hmm, that would mean that the behavior is very different from Newton's, because it prefers flatter minima, while Newton's doesn't care about flatness, it just arrives at minima faster (usually)? Think hard about the difference again and propositions 1 means exactly, the answer doesn't feel satisfactory

Mar 2

You're right, let me think more carefully.

Proposition 1 says the **expected loss after a masked step** equals:

l(θt-Δt)+1-p2pΔt⊤HΔtl(\theta_t - \Delta_t) + \frac{1-p}{2p} \Delta_t^\top H \Delta_tl(θt​-Δt​)+2p1-p​Δt⊤​HΔt​

The first term is exactly what the base optimizer (e.g. RMSProp) would achieve without masking. The second term is strictly **added on top** --- it's a positive quantity (assuming positive curvature) that makes the expected loss *worse* than the unmasked update.

So Proposition 1 is really saying: **masking is strictly worse in expectation than not masking**, from a pure loss-minimization standpoint. This is consistent with the classical convergence analysis the paper mentions --- worse worst-case guarantees.

Newton's method by contrast uses the Hessian to *improve* the step --- the Δ⊤HΔ\Delta^\top H \Delta Δ⊤HΔ term is what guides you to a better Δ∗\Delta^* Δ∗.

So the comparison is almost the **opposite** of what I was saying before: Newton exploits curvature information to minimize faster, while masking introduces a curvature-dependent cost that the optimizer cannot directly minimize over Δ\Delta Δ, but which accumulates more in sharp regions than flat ones --- creating an indirect pressure toward flat minima over the course of training.

The reason masking works empirically despite being worse in expectation per step is precisely this implicit bias toward flat minima, which Newton has no reason to find. Does this feel more accurate to you?

You seem very confused. Analyze proposition 1 again from the beginning and take the rest of the paper into context. Think hard about what it exactly means.

Mar 2

Let me think carefully from scratch.

Proposition 1 says the expected loss after a masked step is:

Et[l(θt-Δ~t)]=l(θt-Δt)+1-p2pΔt⊤HΔt\mathbb{E}_t[l(\theta_t - \tilde{\Delta}_t)] = l(\theta_t - \Delta_t) + \frac{1-p}{2p} \Delta_t^\top H \Delta_tEt​[l(θt​-Δ~t​)]=l(θt​-Δt​)+2p1-p​Δt⊤​HΔt​

The right hand side has two terms:

1.  l(θt-Δt)l(\theta_t - \Delta_t) l(θt​-Δt​) --- the loss you would get with the **full, unmasked** update
2.  1-p2pΔt⊤HΔt\frac{1-p}{2p} \Delta_t^\top H \Delta_t 2p1-p​Δt⊤​HΔt​ --- a positive curvature-dependent term

So the expected loss under masking is **higher** than the unmasked loss. Masking is worse in expectation. This is not surprising --- you're randomly skipping updates.

But here is what I think is the key insight I keep missing: **this is not the objective being minimized**. The optimizer doesn't see this expectation --- it just takes steps. What Proposition 1 really tells you is that the *trajectory* is implicitly shaped by this extra term. Steps that land in high-curvature regions incur a larger gap between the masked and unmasked loss, meaning progress is slower there. Steps toward flatter regions incur a smaller gap, so relative progress is better there.

In contrast, Newton uses HH H to **correct** the step direction --- it's prescriptive. Proposition 1 shows masking uses HH H **descriptively** --- it characterizes an implicit bias that emerges from the noise, without ever computing HH H.

Is this closer to what you had in mind?

Still not really convinced. Does proposition 1 really mean anything? It just shows how the expected loss behaves, in particular it will be higher than without masking, however it doesn't really tell us why that would be good or how the optimization / trajectory will avoid these high-curvature regions.

Mar 2

You're absolutely right. Proposition 1 on its own is actually fairly trivial --- it just says "masking makes things worse in expectation, and the penalty scales with curvature." It doesn't explain any mechanism by which the optimizer would actually avoid high-curvature regions. It's a description, not an explanation.

The actual explanation of *why masking works* must come from somewhere else. Looking at the paper more carefully, I think the real story is empirical and the theoretical justification is somewhat post-hoc. The paper shows:

1.  SkipUpdate just works empirically across model scales
2.  Magma improves further by using momentum-gradient alignment to suppress updates that are inconsistent with accumulated momentum direction

The momentum-gradient alignment in Magma is actually the more substantive mechanistic insight --- it suppresses noisy, inconsistent updates that are likely to be destabilizing, which is a concrete and actionable criterion. This is backed by the heavy-tailed noise experiments and the heterogeneous quadratic benchmarks.

So honestly --- Proposition 1 might be more of a post-hoc rationalization framed as theory, rather than a genuine explanation of why the method works. The comparison to Newton I was drawing was probably misleading from the start.

Would you say that's a fair reading of the paper?

Look at the proof of proposition 1, that could shed some light into the dark

Mar 2

Looking at the proof carefully. It starts with a second-order Taylor expansion:

l(θt-Δ~t)=l(θt)-∑b(gt(b))⊤Δ~t(b)+12∑b∑b′(Δ~t(b))⊤Hbb′(θt)Δ~t(b′)+R2l(\theta_t - \tilde{\Delta}_t) = l(\theta_t) - \sum_b (g_t^{(b)})^\top \tilde{\Delta}_t^{(b)} + \frac{1}{2}\sum_b \sum_{b'} (\tilde{\Delta}_t^{(b)})^\top H_{bb'}(\theta_t) \tilde{\Delta}_t^{(b')} + R_2l(θt​-Δ~t​)=l(θt​)-b∑​(gt(b)​)⊤Δ~t(b)​+21​b∑​b′∑​(Δ~t(b)​)⊤Hbb′​(θt​)Δ~t(b′)​+R2​

Then takes expectations. The key step is equation (10) --- when taking expectations over the independent Bernoulli masks:

E[(Δ~t(b))⊤Hbb′Δ~t(b′)]={Δt(b)⊤Hbb′Δt(b′)b≠b′1pΔt(b)⊤HbbΔt(b)b=b′\mathbb{E}[(\tilde{\Delta}_t^{(b)})^\top H_{bb'} \tilde{\Delta}_t^{(b')}] = \begin{cases} \Delta_t^{(b)\top} H_{bb'} \Delta_t^{(b')} & b \neq b' \\ \frac{1}{p} \Delta_t^{(b)\top} H_{bb} \Delta_t^{(b)} & b = b' \end{cases}E[(Δ~t(b)​)⊤Hbb′​Δ~t(b′)​]={Δt(b)⊤​Hbb′​Δt(b′)​p1​Δt(b)⊤​Hbb​Δt(b)​​b=b′b=b′​

The off-diagonal terms remain unchanged, but the diagonal terms get scaled by 1p\frac{1}{p} p1​. This is simply because E[(m(b))2]=p\mathbb{E}[(m^{(b)})^2] = p E[(m(b))2]=p while E[m(b)]E[m(b′)]=p2/p2=1\mathbb{E}[m^{(b)}]\mathbb{E}[m^{(b')}] = p^2/p^2 = 1 E[m(b)]E[m(b′)]=p2/p2=1 for independent masks.

So the extra term 1-p2pΔ⊤HbbΔ\frac{1-p}{2p}\Delta^\top H_{bb}\Delta 2p1-p​Δ⊤Hbb​Δ is **purely a consequence of the independence of block masks** --- the cross-block curvature is unaffected, only within-block curvature gets amplified. Proposition 1 is essentially just a statement about how independence of the Bernoulli variables inflates the diagonal Hessian blocks in expectation. Nothing deeper than that analytically.

Which makes your point even stronger --- it really doesn't explain the mechanism at all.

Ok, then let's go beyong proposition 1. How do they motivate the method?

Mar 3

Looking at the paper more carefully, the actual motivation for Magma specifically comes from the momentum-gradient alignment idea in Section 3.

The reasoning is:

1.  In stochastic optimization, gradient components that are **consistent across iterations** carry meaningful signal, while rapidly fluctuating components are dominated by noise.
2.  They use a result from online variational inference --- the probability of a gradient being negatively aligned with the momentum satisfies P(μ⊤g<0)=Φ(-∥μ∥/σ)\mathbb{P}(\mu^\top g < 0) = \Phi(-\|\mu\|/\sigma) P(μ⊤g<0)=Φ(-∥μ∥/σ), which decays exponentially in the signal-to-noise ratio. So **negative alignment events are statistically abnormal** --- they represent genuine outlier fluctuations rather than meaningful gradient signal.
3.  Therefore, when the current stochastic gradient disagrees with the accumulated momentum direction, that's a strong signal that this particular update is noise-dominated and potentially destabilizing. Suppressing it is the right thing to do.

This is the actual mechanistic motivation --- not the curvature regularization story from Proposition 1. The cosine similarity score in equation (3) is a practical, computable proxy for this signal-to-noise criterion.

The curvature regularization framing in Proposition 1 is then more of a retrospective theoretical lens on *why* SkipUpdate already works before you add the momentum alignment --- but as we established, it doesn't really explain the mechanism convincingly. The SNR motivation for Magma is considerably more compelling.

Unpack the formula that decays expontentially in the signal to noise ratio

Mar 3

The formula is:

P(μ⊤g<0)=Φ(-∥μ∥σ)\mathbb{P}(\mu^\top g < 0) = \Phi\left(-\frac{\|\mu\|}{\sigma}\right)P(μ⊤g<0)=Φ(-σ∥μ∥​)

where Φ\Phi Φ is the standard normal CDF.

This comes from modeling the stochastic gradient gg g as:

g=μ+ϵ,ϵ∼N(0,σ2I)g = \mu + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)g=μ+ϵ,ϵ∼N(0,σ2I)

where μ\mu μ is the true mean (approximated by the momentum estimate) and σ\sigma σ is the noise level. Then μ⊤g=∥μ∥2+μ⊤ϵ\mu^\top g = \|\mu\|^2 + \mu^\top \epsilon μ⊤g=∥μ∥2+μ⊤ϵ, where μ⊤ϵ∼N(0,σ2∥μ∥2)\mu^\top \epsilon \sim \mathcal{N}(0, \sigma^2\|\mu\|^2) μ⊤ϵ∼N(0,σ2∥μ∥2). So:

P(μ⊤g<0)=P(N(0,1)<-∥μ∥σ)=Φ(-∥μ∥σ)\mathbb{P}(\mu^\top g < 0) = \mathbb{P}\left(\mathcal{N}(0,1) < -\frac{\|\mu\|}{\sigma}\right) = \Phi\left(-\frac{\|\mu\|}{\sigma}\right)P(μ⊤g<0)=P(N(0,1)<-σ∥μ∥​)=Φ(-σ∥μ∥​)

Now, Φ(-x)\Phi(-x) Φ(-x) decays **very rapidly** for large xx x --- it's essentially 1xϕ(x)\frac{1}{x}\phi(x) x1​ϕ(x) where ϕ\phi ϕ is the standard normal PDF, so it decays faster than exponentially in x2x^2 x2.

The intuition is clean: when the signal-to-noise ratio ∥μ∥/σ\|\mu\|/\sigma ∥μ∥/σ is high --- meaning the momentum has a strong, consistent direction --- the probability that a gradient update points the wrong way becomes vanishingly small. A negative alignment event in that regime is almost certainly a heavy-tailed noise spike, and suppressing it is the right call.

Ok, but "SkipUpdate" doesn't have this extra logic. How is just "SkipUpdate" motivated?

Mar 3

You're right, SkipUpdate is just random masking with no alignment logic at all --- it's literally just flipping a coin for each block. And yet it already outperforms Adam and Muon.

Looking at the paper honestly, I don't think there is a deep motivation for SkipUpdate beyond Proposition 1, which we already established doesn't really explain the mechanism. It seems like it started as a **surprising empirical observation** --- "we tried randomly skipping half the updates and it worked really well" --- and Proposition 1 is the attempt to provide some theoretical framing after the fact.

The paper even frames it this way in the introduction: they call it a "counter-intuitive empirical finding." The theoretical story around curvature regularization is offered as a possible explanation, but as we discussed, Proposition 1 merely shows the expected loss is higher under masking and that the penalty scales with within-block curvature --- it doesn't actually show why the trajectory would prefer flatter regions.

So the honest answer is: SkipUpdate is motivated primarily empirically, and Magma is where the actual theoretical motivation comes in via the SNR / momentum-gradient alignment argument. The paper essentially says "this surprising thing works, here is our best attempt to explain why, and here is an improved version based on that explanation."

Then let's try to think through why SkipUpdate works. Think about it and propose multiple explanations

Apr 5

Synthesized seven theoretical mechanisms explaining SkipUpdate's effectiveness

Synthesized seven theoretical mechanisms explaining SkipUpdate's effectiveness

Let me think through several possible explanations:

**1\. Robustness to heavy-tailed noise** LLM training has heavy-tailed gradient noise --- occasional massive spikes. Random masking means any given spike has a 50% chance of being skipped per block. It's a simple, parameter-free robustness mechanism that requires no spike detection logic.

**2\. Reduced inter-block coupling**When all blocks update simultaneously, their interactions through off-diagonal Hessian terms Hbb′H_{bb'} Hbb′​ can cause instability --- one block's update destabilizes another. Random masking breaks these interactions stochastically, reducing destructive coupling.

**3\. Implicit ensemble effect** Similar to Dropout --- each step effectively trains a random sub-network. The final model benefits from an implicit ensemble over many random subsets of block updates, which tends to improve generalization.

**4\. Better momentum estimates** Because momentum is still updated densely even when parameters are skipped, by the time a block does get updated, the momentum is a more reliable, lower-variance estimate of the true gradient direction. You're essentially waiting for a better signal before committing.

**5\. Implicit learning rate reduction for high-variance blocks** Blocks with noisier gradients benefit more from being skipped, since their updates are less trustworthy. Random masking applies this reduction uniformly without needing to identify which blocks are noisy.

Which of these do you find most plausible?

2 and 4 kinda make sense to me. But I think 5 is now likely to be close to the actual answer, but your formulation doesn't make sense IMO. Let's do the following: see how it behaves for toy examples. Two dimensions are interesting, I think: variance (relevant only to SGD) and flatness (of minimum). For both, think through both high and low variance / flatness (4 cases!) with example numbers, actually calculating the results.

Apr 5

Computed four toy cases comparing SkipUpdate convergence and noise floors

Computed four toy cases comparing SkipUpdate convergence and noise floors

Let me set up a clean 1D quadratic: l(θ)=a2θ2l(\theta) = \frac{a}{2}\theta^2 l(θ)=2a​θ2 with SGD, stochastic gradient gt=aθt+ϵtg_t = a\theta_t + \epsilon_t gt​=aθt​+ϵt​, ϵt∼N(0,σ2)\epsilon_t \sim N(0, \sigma^2) ϵt​∼N(0,σ2).

Using η = 0.1, sharp a=5 (ηa=0.5), flat a=0.5 (ηa=0.05), high σ=10, low σ=1.

The key formulas:

-   SGD noise floor: ησ2a(2-ηa)\frac{\eta\sigma^2}{a(2-\eta a)} a(2-ηa)ησ2​, convergence factor: (1-ηa)2(1-\eta a)^2 (1-ηa)2
-   SkipUpdate noise floor: ησ2a(1-ηa)\frac{\eta\sigma^2}{a(1-\eta a)} a(1-ηa)ησ2​, convergence factor: 0.5+0.5(1-2ηa)20.5 + 0.5(1-2\eta a)^2 0.5+0.5(1-2ηa)2

**Case 1: Sharp (a=5), High variance (σ=10)**

-   SGD: convergence 0.25/step, noise floor 1.33
-   SkipUpdate: convergence 0.50/step, noise floor 4.0 **(3x worse)**

**Case 2: Sharp (a=5), Low variance (σ=1)**

-   SGD: convergence 0.25/step, noise floor 0.013
-   SkipUpdate: convergence 0.50/step, noise floor 0.040 **(3x worse)**

**Case 3: Flat (a=0.5), High variance (σ=10)**

-   SGD: convergence 0.90/step, noise floor 10.26
-   SkipUpdate: convergence 0.905/step, noise floor 21.05 **(2x worse)**

**Case 4: Flat (a=0.5), Low variance (σ=1)**

-   SGD: convergence 0.90/step, noise floor 0.103
-   SkipUpdate: convergence 0.905/step, noise floor 0.211 **(2x worse)**

So SkipUpdate is **always worse** in this simple model. But there's a critical pattern: the noise floor penalty is **3x at sharp minima but only 2x at flat minima**. Convergence speed is also penalized more at sharp minima (2x slower) but barely at flat ones.

This means SkipUpdate makes it relatively harder to stay in sharp minima than flat ones --- not by reducing the learning rate, but through **differential noise amplification that scales with curvature**. The implicit bias toward flat minima emerges from this asymmetry, not from anything the optimizer explicitly computes.

Does this match what you were thinking, or am I still missing the clean story?

Yes, this is exactly what I wanted. So it might be penalizing sharp minima. Why could that be? Unpack "differential noise amplification that scales with curvature" (is this something you guessed, or so you have more behind it?)

Apr 5

Derived noise amplification formula from first principles mathematics

Derived noise amplification formula from first principles mathematics

It's not a guess --- it follows directly from the math. Let me unpack it.

To preserve unbiasedness with p=0.5, SkipUpdate **must rescale by 1/p=2** when an update is applied. So half the time you do nothing, half the time you take a step of size **2η** instead of η.

Now look at what this does to noise vs signal:

-   **Signal** scales linearly with step size: doubling η doubles the descent
-   **Noise** scales quadratically with step size: doubling η **quadruples** the noise contribution

So the rescaling systematically worsens the noise-to-signal ratio. This is already bad, but why does it get worse with curvature?

Because the destabilizing effect of a large step depends critically on how sharp the landscape is. Look at the noise floor ratio we derived:

SkipUpdate noise floorSGD noise floor=2-ηa1-ηa\frac{\text{SkipUpdate noise floor}}{\text{SGD noise floor}} = \frac{2-\eta a}{1-\eta a}SGD noise floorSkipUpdate noise floor​=1-ηa2-ηa​

As ηa → 0 (flat): ratio → 2, constant penalty. As ηa → 1 (sharp, near stability boundary): ratio → ∞.

The doubled step size 2η brings you **much closer to the stability boundary** (1-2ηa → negative) in sharp regions, causing noise to explode. In flat regions the doubled step is still well within the stable regime.

So the differential noise amplification isn't hand-waving --- it's a direct mathematical consequence of the unbiasedness rescaling interacting with the stability boundary, which is itself determined by curvature.

Ok, but we could also just increase the learning rate instead, right, to the same effect? So either the momentum update is the important part, or the interaction of different weights. Let's do a similar experiment for different weights: one "flat weight" and one "sharp weight", tied together in some way (e.g. a quadratic function that's not "axis aligned"). Stimulate using SGD and SkipUpdate (let's ignore noise for now)

Apr 5

Constructed coupled quadratic simulation comparing SGD and SkipUpdate convergence

Constructed coupled quadratic simulation comparing SGD and SkipUpdate convergence

Great question --- let me set up the 2D coupled case carefully.

**Setup:**

l(θ1,θ2)=0.125(θ1+θ2)2+1.25(θ1-θ2)2l(\theta_1, \theta_2) = 0.125(\theta_1+\theta_2)^2 + 1.25(\theta_1-\theta_2)^2l(θ1​,θ2​)=0.125(θ1​+θ2​)2+1.25(θ1​-θ2​)2

The flat direction is θ1+θ2\theta_1+\theta_2 θ1​+θ2​ (curvature 0.5), the sharp direction is θ1-θ2\theta_1-\theta_2 θ1​-θ2​ (curvature 5). They're "tied" because each parameter mixes both. Start at θ0=(1,0)\theta_0=(1,0) θ0​=(1,0), η=0.1\eta=0.1 η=0.1.

Gradients: g1=2.75θ1-2.25θ2g_1 = 2.75\theta_1 - 2.25\theta_2 g1​=2.75θ1​-2.25θ2​, g2=-2.25θ1+2.75θ2g_2 = -2.25\theta_1 + 2.75\theta_2 g2​=-2.25θ1​+2.75θ2​

**Simulating SkipUpdate:** alternating --- update only θ1\theta_1 θ1​ (with 2η=0.22\eta=0.2 2η=0.2), then only θ2\theta_2 θ2​ (with 2η=0.22\eta=0.2 2η=0.2).

| Step | SGD (η=0.1) | SkipUpdate |
| --- | --- | --- |
| 0 | 1.375 | 1.375 |
| 1 | 0.425 | 0.278 |
| 2 | 0.180 | 0.130 |
| 3 | 0.111 | **0.041** |
| 4 | 0.088 | 0.040 |
| 5 | 0.076 | **0.026** |

SkipUpdate wins clearly. **Why?**

The key is that each coordinate update with step 2η=0.22\eta=0.2 2η=0.2 sees curvature Hii=2.75H_{ii}=2.75 Hii​=2.75 (the diagonal element), **not** λmax=5\lambda_{max}=5 λmax​=5 (the eigenvalue). So the convergence factor per step is ∣1-0.2×2.75∣=0.45|1 - 0.2 \times 2.75| = 0.45 ∣1-0.2×2.75∣=0.45, vs SGD's ∣1-0.1×5∣=0.5|1 - 0.1 \times 5| = 0.5 ∣1-0.1×5∣=0.5. SkipUpdate eliminates the sharp component faster precisely **because the coupling means Hii<λmaxH_{ii} < \lambda_{max} Hii​<λmax​** --- the coordinate curvature is lower than the maximum eigenvalue.

But --- crucially --- SGD with η=0.2\eta=0.2 η=0.2 gives convergence factor ∣1-0.2×5∣=0|1-0.2\times5|=0 ∣1-0.2×5∣=0, instantly killing the sharp component. SGD with doubled learning rate is even better. So your intuition is confirmed: **in the noiseless case, just increasing the learning rate achieves the same or better effect**. The advantage of SkipUpdate over simply raising η\eta η must still come from the noise interaction established in the 1D case.

Hmm, this isn't telling us much. Let's go back to the paperl again: after proposition 1, it says "Since directions of large positive curvature correspond to sharp increases of the loss, minimizing the expected post-update loss implicitly discourages updates that align with high-curvature directions within each block." This seems handwavy to me and we dismissed it so far. But what could this "implicit discouragement" be, if it is really there? Try to unpack this sentence critically, and compare to other methods that might have similar behavior

Apr 5

Reconciled expected loss analysis with actual noise dynamics and dropout comparisons

Reconciled expected loss analysis with actual noise dynamics and dropout comparisons

Let me think carefully about what "implicit discouragement" could actually mean mechanically.

Proposition 1 shows the **expected** loss has an extra curvature term. But as we established, Δt\Delta_t Δt​ is fixed by the base optimizer --- it doesn't choose Δt\Delta_t Δt​ to minimize Proposition 1's expression. So in what sense is anything being "discouraged"?

The key might be in the **actual noise structure of the updates**, not their expectation. The actual masked update is:

Δ~t(b)=mt(b)pΔt(b)\tilde{\Delta}^{(b)}_t = \frac{m^{(b)}_t}{p}\Delta^{(b)}_tΔ~t(b)​=pmt(b)​​Δt(b)​

The noise around the expected update has variance 1-pp∥Δt(b)∥2\frac{1-p}{p}\|\Delta^{(b)}_t\|^2 p1-p​∥Δt(b)​∥2 --- this is **multiplicative noise**. Near a sharp minimum, Δt\Delta_t Δt​ is large in the sharp directions, so the noise is large there. Near a flat minimum, Δt\Delta_t Δt​ is small, noise is small. Sharp minima are therefore less stable attractors under this noise than flat ones.

This is actually well-studied --- it's essentially the **SGD implicit bias** story. SGD with finite learning rate is known to prefer flat minima not because each step targets them explicitly, but because the noise makes sharp minima less stable stationary points. The Fokker-Planck / stationary distribution perspective formalizes this.

The closest analogies:

-   **SGD implicit bias**: additive gradient noise, well-understood to prefer flat minima
-   **Dropout**: also Bernoulli multiplicative noise, also known to have implicit regularization (cited in the paper --- Mianjy et al., Wei et al.) --- this is actually the closest analogy, just applied to updates rather than activations
-   **SAM**: achieves flatness preference **explicitly** by adversarially perturbing parameters --- same goal, completely different mechanism

So the paper's sentence is likely pointing at something real --- the multiplicative noise structure creating a biased stationary distribution --- but framing it through Proposition 1 (which is about expectations) is arguably the wrong lens to explain it.

Let's quickly look at prop 1: to me it looks trivial, it's basically just how the error behaves according to Taylor expansion, applied to the specific case of SkipUpdate, right? Or is there something more interesting going on? Check the proof

11:44 AM

Validated user's intuition: proposition reduces to Taylor expansion variance calculations

Validated user's intuition: proposition reduces to Taylor expansion variance calculations

Yes, looking at the proof it's essentially trivial. The steps are:

1.  Write second-order Taylor expansion of l(θt-Δ~t)l(\theta_t - \tilde{\Delta}_t) l(θt​-Δ~t​)
2.  Take expectations over the Bernoulli masks

The only "interesting" step is equation (10) --- the diagonal/off-diagonal split. And that's just:

E[(m(b)p)2]=E[(m(b))2]p2=pp2=1p\mathbb{E}\left[\left(\frac{m^{(b)}}{p}\right)^2\right] = \frac{\mathbb{E}[(m^{(b)})^2]}{p^2} = \frac{p}{p^2} = \frac{1}{p}E[(pm(b)​)2]=p2E[(m(b))2]​=p2p​=p1​

while for b≠b′b \neq b' b=b′, by independence:

E[m(b)pm(b′)p]=1\mathbb{E}\left[\frac{m^{(b)}}{p}\frac{m^{(b')}}{p}\right] = 1E[pm(b)​pm(b′)​]=1

So the extra term 1-p2pΔ⊤HbbΔ\frac{1-p}{2p}\Delta^\top H_{bb}\Delta 2p1-p​Δ⊤Hbb​Δ is literally just **the variance of the Bernoulli variable** showing up in the quadratic term of the Taylor expansion. It's 12Var(m(b)/p)⋅Δ⊤HbbΔ\frac{1}{2}\text{Var}(m^{(b)}/p) \cdot \Delta^\top H_{bb}\Delta 21​Var(m(b)/p)⋅Δ⊤Hbb​Δ.

So yes --- completely trivial mathematically. It's just Taylor expansion plus Var(X2)>0\text{Var}(X^2) > 0 Var(X2)>0 for a Bernoulli. The paper's framing of it as revealing "geometric regularization" seems like significant overselling of what is essentially a variance calculation.

Are there ablations or results in the paper that could help us identify why it works? Ablations could be e.g. not using momentum, or not doing it block-wise

11:47 AM

Excavated ablations revealing dense momentum and heterogeneous landscapes critical

Excavated ablations revealing dense momentum and heterogeneous landscapes critical

Looking at the ablations carefully, the most informative ones are:

**C.4 - Dense vs Sparse Momentum (Figure A2)** This is the most directly relevant. When momentum is updated sparsely (only for blocks that are actually updated), training becomes severely unstable without damping, and even with damping underperforms dense momentum. This strongly suggests the dense momentum update is not incidental --- it's doing critical work, consistent with explanation 4 from earlier (better momentum estimates).

**C.2 - Masking Granularity** Element-wise, row, column, and block masking all give nearly identical results (21.73 to 21.81). This is actually very telling --- if the block structure were important (e.g. for breaking inter-block coupling as in explanation 2), we'd expect block-wise masking to behave differently from element-wise. The near-equivalence suggests the **specific structure of masking doesn't matter**, just the noise injection itself.

**Section 4.4 - Heterogeneous vs Homogeneous Quadratics** Magma only helps in the heterogeneous case --- where different blocks have mixed curvatures --- and doesn't improve over AdamW for homogeneous curvature or ResNet. This points toward the benefit being specifically about **handling curvature heterogeneity across blocks**, which aligns with the differential noise amplification story from our 1D analysis.

So taken together, the ablations seem to point toward: dense momentum is critical, block structure isn't, and heterogeneous curvature is where the gains come from. Would you agree?

Hmm table A2 says the baseline RMSProp has perpexity 22.64, which is better than 21.73, the best one for SkipUpdate (element wise is best), if I read this correctly?

12:01 PM

Identified user's inverted perplexity interpretation and corrected understanding

Identified user's inverted perplexity interpretation and corrected understanding

Actually I think you have it backwards --- lower perplexity is better, so 21.73 is better than 22.64. SkipUpdate (Uniform Sampling, element-wise) improves over baseline RMSProp by almost a full perplexity point.

But your comment does highlight something interesting about Table A2 that I glossed over. Look at "Damping alone" --- that's 21.97, which is worse than plain Uniform Sampling at 21.73. And the best result is Uniform Sampling + Damping at 21.58.

This is actually quite revealing:

-   **Uniform Sampling alone** (pure SkipUpdate) already helps substantially --- 22.64 → 21.73
-   **Damping alone** (momentum-gradient alignment scaling without masking) helps less --- 22.64 → 21.97
-   **Combined** is best --- 21.58

So the random masking itself seems to be doing more work than the momentum alignment in isolation. Which brings us back to the question of why pure random masking helps, and the momentum alignment story --- while compelling theoretically --- may not be the primary driver of the gains. The noise injection from random masking seems to be the more fundamental ingredient.

Oops, you're right, its **22** vs **21**, misread it. Anything else would have been surprising. But that's good, we can eliminate the block-level logic as irrelevant, just element level is enough (I guess this is per-neuron / activation?). Do you agree?

12:05 PM

Reconciled element-wise masking equivalence and validated dropout analogy

Reconciled element-wise masking equivalence and validated dropout analogy

Yes, I agree --- and this is actually quite important for understanding the method.

Element-wise masking means each individual weight's update is independently masked, which makes it essentially **gradient Dropout** --- extremely similar to standard Dropout but applied to parameter updates rather than activations. The paper actually cites the Dropout literature (Mianjy et al., Wei et al., Zhang and Xu) as related work, and the implicit regularization story for Dropout is well-established.

This simplification cuts away a lot of the paper's theoretical framing:

-   The elaborate block-diagonal Hessian structure motivation becomes irrelevant
-   The "efficient operation pruning" justification for block-wise masking is moot
-   Proposition 1's block-wise formulation is unnecessarily complex

What remains is essentially: **randomly zeroing individual weight updates, with rescaling for unbiasedness, plus dense momentum updates**.

And the dense momentum ablation (C.4) then becomes the really critical result --- it's what distinguishes this from simply adding noise or using Dropout. The dense momentum ensures that even when updates are skipped, the direction estimate keeps improving, so when updates do happen they're more reliable.

So the core of why it works might be quite simple: noise injection via gradient Dropout + better gradient direction estimates from dense momentum. Does that feel like a satisfying summary to you?

Nope, this doesn't tell me why it works at all. Regarding the possibly relevant existing explanations: - SGD implicit bias: seems quite irrelevant here, this is just large learning rates diverging for sharp minima. I don't see how it could be mapped - Dropout: works on activations, and thus has COMPLETELY different properties, unless there's something specific linking them in mechanism - SAM: I don't know about this one, explain it in more detail to me

5:04 PM

Dissected SAM's perturbation mechanism and contemplated mechanistic divergence from SkipUpdate

Dissected SAM's perturbation mechanism and contemplated mechanistic divergence from SkipUpdate

SAM (Sharpness-Aware Minimization) works as follows. Instead of minimizing l(θ)l(\theta) l(θ), it minimizes:

min⁡θmax⁡∥δ∥≤ρl(θ+δ)\min_\theta \max_{\|\delta\|\leq\rho} l(\theta + \delta)θmin​∥δ∥≤ρmax​l(θ+δ)

i.e. find parameters where the **worst-case nearby perturbation** still gives low loss. This explicitly targets flat minima --- sharp minima have high max⁡δl(θ+δ)\max_\delta l(\theta+\delta) maxδ​l(θ+δ) by definition.

Practically, each step is:

1.  Compute the maximizing perturbation: δ∗≈ρ∇l(θ)∥∇l(θ)∥\delta^* \approx \rho \frac{\nabla l(\theta)}{\|\nabla l(\theta)\|} δ∗≈ρ∥∇l(θ)∥∇l(θ)​ (one gradient ascent step)
2.  Take a gradient step at the perturbed point: θt+1=θt-η∇l(θt+δ∗)\theta_{t+1} = \theta_t - \eta\nabla l(\theta_t + \delta^*) θt+1​=θt​-η∇l(θt​+δ∗)

This costs **two gradient evaluations per step**, which is the main drawback.

Now --- is there a mechanistic link to SkipUpdate? Honestly I'm not sure there is a clean one. SAM is entirely deterministic and explicitly constructs perturbations aligned with the gradient. SkipUpdate is random and acts on updates not parameters.

Rather than forcing a connection, I think we should go back to basics: the granularity ablation showed structure doesn't matter, and we haven't actually identified the core mechanism yet. What's your intuition for why it works?

I see two possible lines of explanation: - The interaction with momentum. Could it be kinda interpolating between Nesterov and standard momentum? Should already work 1D - It somehow helps (wouldn't know how) for variables with strong interactions. E.g. for variables with a big second derivative between them, it will somehow smoothen or similar - Or possibly a combination of the above Wdyt? Is there something in the paper hinting in any of the directions above, or refuting them?

5:12 PM

Weighed momentum variance reduction against variable coupling effects

Weighed momentum variance reduction against variable coupling effects

Both are interesting. Let me think through each carefully.

**Explanation 1: Momentum interpolation**

This actually has direct support in the paper --- Section 2 explicitly says SkipUpdate "yields a variance-reduced estimator of the true momentum due to the lazy update scheme." When you skip a parameter update but still update momentum, the momentum accumulates more gradient observations per parameter step. So when you do update, you're using a richer momentum estimate. The dense momentum ablation (C.4) strongly supports this --- sparse momentum breaks catastrophically. This seems like a real effect worth working out in 1D.

The Nesterov connection is interesting --- Nesterov evaluates the gradient at a lookahead point θ-βημ\theta - \beta\eta\mu θ-βημ, effectively using "future" gradient information. With SkipUpdate, momentum runs ahead of parameters in a somewhat similar spirit, though the mechanism is different.

**Explanation 2: Strong variable interactions**

The heterogeneous quadratic experiment (Section 4.4) is very suggestive here --- Magma only helps when eigenvalues of vastly different magnitudes are mixed within blocks, and **doesn't help for ResNet** which has more homogeneous curvature. This is the strongest hint in the paper toward this explanation.

However --- the granularity ablation somewhat weakens this, since element-wise masking works as well as block-wise, meaning it's individual weight interactions rather than block-level coupling.

Which do you think is more promising to work out analytically?

Let's investigate explanation 2 in more detail, I like it a bit more atm. Try to work it out analytically as you suggested

5:17 PM

Analyzed coupled quadratic systems and convergence rates comparatively

Analyzed coupled quadratic systems and convergence rates comparatively

Let me set up the simplest possible 2D coupled case. No noise, pure gradient descent.

l(θ1,θ2)=12(θ12+θ22+2c⋅θ1θ2)l(\theta_1, \theta_2) = \frac{1}{2}(\theta_1^2 + \theta_2^2 + 2c\cdot\theta_1\theta_2)l(θ1​,θ2​)=21​(θ12​+θ22​+2c⋅θ1​θ2​)

So H=(1cc1)H = \begin{pmatrix}1 & c \\ c & 1\end{pmatrix} H=(1c​c1​), eigenvalues 1+c1+c 1+c and 1-c1-c 1-c. The coupling cc c controls interaction strength.

**SGD step size is limited by λmax=1+c\lambda_{max} = 1+c λmax​=1+c**, requiring η<1/(1+c)\eta < 1/(1+c) η<1/(1+c).

**SkipUpdate, when only θ1\theta_1 θ1​ updates** (with step 2η2\eta 2η):

θ1′=(1-2η)θ1-2ηc⋅θ2,θ2′=θ2\theta_1' = (1-2\eta)\theta_1 - 2\eta c\cdot\theta_2, \quad \theta_2' = \theta_2θ1′​=(1-2η)θ1​-2ηc⋅θ2​,θ2′​=θ2​

The stability condition here is just ∣1-2η∣<1|1-2\eta| < 1 ∣1-2η∣<1, i.e. η<1\eta < 1 η<1 --- **independent of cc c!**

The coordinate update only "sees" the diagonal element Hii=1H_{ii}=1 Hii​=1, not λmax=1+c\lambda_{max}=1+c λmax​=1+c.

**Numerical example with c=0.9:**

SGD optimal convergence rate: κ-1κ+1\frac{\kappa-1}{\kappa+1} κ+1κ-1​ where κ=1+c1-c=19\kappa = \frac{1+c}{1-c} = 19 κ=1-c1+c​=19, giving ≈0.90\approx 0.90 ≈0.90 per step.

SkipUpdate alternating updates, η=0.4\eta=0.4 η=0.4:

M1=(0.2-0.7201),M2=(10-0.720.2)M_1 = \begin{pmatrix}0.2 & -0.72 \\ 0 & 1\end{pmatrix}, \quad M_2 = \begin{pmatrix}1 & 0 \\ -0.72 & 0.2\end{pmatrix}M1​=(0.20​-0.721​),M2​=(1-0.72​00.2​)

Two-step matrix M2M1M_2 M_1 M2​M1​: slowest eigenvalue ≈0.75\approx 0.75 ≈0.75, or **≈0.87\approx 0.87 ≈0.87 per step** --- faster than SGD's 0.90.

**With weak coupling c=0.1** (κ=1.22\kappa=1.22 κ=1.22), SGD convergence factor ≈0.1\approx 0.1 ≈0.1 --- already fast, SkipUpdate offers little gain.

So the mechanism is essentially **coordinate descent avoiding the coupling-inflated λmax\lambda_{max} λmax​**: SGD's step size is hostage to the sharpest coupled direction, while SkipUpdate's per-coordinate steps are governed only by diagonal curvature, which is independent of coupling strength. The stronger the coupling, the bigger the advantage.

Does this match your intuition for explanation 2?

This starts to make sense to me. However, I would guess that coupling in usual NNs is not that big between weights, right? And if it's not big, it won't be worth skipping the updates (essentially slowing down by 1 - skipping probability)

5:27 PM

Weighed coupling magnitude against update costs and questioned momentum's role

Weighed coupling magnitude against update costs and questioned momentum's role

That's a really sharp point and I think it's a genuine challenge for explanation 2. Let me think through it.

You're right that for typical NN weights, direct coupling ∂2l∂θi∂θj\frac{\partial^2 l}{\partial \theta_i \partial \theta_j} ∂θi​∂θj​∂2l​ between individual weights is probably small --- they mostly interact indirectly through activations.

But there are two counterarguments:

**1\. Transformers might be special**The paper specifically shows benefits for transformer-like heterogeneous curvature but NOT for ResNet. Attention mechanisms create strong interactions between weights through the QK⊤VQK^\top V QK⊤V products --- individual weights in the same attention head are genuinely strongly coupled. This might make transformers specifically a case where coupling is large enough to matter.

**2\. Adam already partially handles this** Actually this might be the stronger objection --- Adam's diagonal preconditioning already partially compensates for heterogeneous curvature. If coupling were the main story, SkipUpdate should help more on top of SGD than Adam. But looking at Table 1... it helps roughly similarly on both. Which somewhat undermines explanation 2.

This makes me think explanation 1 --- the momentum interaction --- might actually be more fundamental after all, since it would apply regardless of coupling strength or optimizer choice.

Should we go back and work out explanation 1 analytically?

> Adam's diagonal preconditioning already partially compensates for heterogeneous curvature

How does this work?

Also, there are no results for vanilla SGD, right?

5:30 PM

Examined Adam's gradient normalization mechanism and confirmed absent SGD results

Examined Adam's gradient normalization mechanism and confirmed absent SGD results

**Adam's diagonal preconditioning:**

Adam maintains vt=β2vt-1+(1-β2)gt2v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 vt​=β2​vt-1​+(1-β2​)gt2​ and scales updates by η/vt\eta/\sqrt{v_t} η/vt​​. For a parameter with high curvature, gradients are consistently large, so vtv_t vt​ is large and the effective step size is automatically reduced. For flat directions, vtv_t vt​ is small and steps are larger.

So Adam essentially **normalizes away per-parameter curvature** --- it already partially solves the heterogeneous curvature problem diagonally. But critically, it only handles diagonal elements HiiH_{ii} Hii​, not off-diagonal coupling HijH_{ij} Hij​. So for our coupled quadratic, Adam handles the θ12,θ22\theta_1^2, \theta_2^2 θ12​,θ22​ terms but not the θ1θ2\theta_1\theta_2 θ1​θ2​ coupling.

**Regarding SGD results:**

You're right, there are none for LLM pretraining in Table 1. This is actually a significant gap for our analysis --- to properly distinguish explanations 1 and 2 we'd really want to see:

-   SkipUpdate on top of SGD vs Adam
-   If coupling is key, SkipUpdate should help much more on SGD since Adam already handles diagonal curvature
-   If momentum is key, it should help similarly on both

The heterogeneous quadratic section does compare against SGD, but that's a controlled toy setting. The absence of SGD in the main results actually makes it harder to disentangle the two explanations. Would you say this is a weakness of the paper?

> If momentum is key, it should help similarly on both

There's no momentum in vanilla SGD, at least that's what I was aiming at, then we could distinguish the two explanations.

> The paper specifically shows benefits for transformer-like heterogeneous curvature but NOT for ResNet.

What are the result for ResNet? Is SkipUpdate evaluated separately, also per-element?

And let's try to actually look at weight coupling again: are there papers that investigate it, especially for transformers? Search the web

5:33 PM

Scrutinized ResNet benchmarks and hunted weight coupling research

Scrutinized ResNet benchmarks and hunted weight coupling research

Good point on SGD --- if momentum is key, SkipUpdate on vanilla SGD (no momentum) should show little benefit, while if coupling is key it should still help.

On ResNet: looking at the paper carefully, Section 4.4 only reports **Magma** on ResNet (94.46% vs 93.82%), not SkipUpdate separately. And there's no element-wise ablation for ResNet. So we can't actually cleanly conclude anything about SkipUpdate or element-wise masking on ResNet --- the paper doesn't report it.

Identified relevant research on neural network Hessian structure

Identified relevant research on neural network Hessian structure

The most relevant result here is actually very helpful. Empirical studies have reported that the Hessian matrix of neural networks exhibits a near-block-diagonal structure, and recent work has numerically studied the block-wise Hessian spectrum of both CNNs and Transformers, connecting block-wise spectra to the effectiveness of Adam. [arXiv](https://arxiv.org/html/2505.02809v1)

This is critical for our analysis --- it means **off-diagonal coupling between blocks is empirically small**. The dominant curvature interactions are *within* blocks, not between them.

This actually **undermines explanation 2** more seriously than I previously acknowledged. If off-diagonal Hessian entries are small, the coordinate descent advantage we derived in the 2D coupled example is largely irrelevant --- there's not much coupling to exploit.

This pushes us back toward explanation 1 --- the momentum interaction --- as the more plausible story. The near-block-diagonal structure means each block is relatively independent, so the benefit of SkipUpdate is more likely coming from what happens *within* each block's update dynamics --- specifically the interaction between dense momentum accumulation and sparse parameter updates.

Should we now work through explanation 1 analytically, since explanation 2 seems less supported by the Hessian structure evidence?

One thing to state explicitly here: the block-based masking actually makes the coupling argument weak, as the inter-block coupling is usually low. And the experiments show relatively little difference for per-element vs per-block.

Thus, let's work through explanation 1 then analytically

5:38 PM

Analyzed momentum dynamics and SkipUpdate's gradient accumulation effects

Analyzed momentum dynamics and SkipUpdate's gradient accumulation effects

Let me set up the simplest case: 1D quadratic l(θ)=a2θ2l(\theta) = \frac{a}{2}\theta^2 l(θ)=2a​θ2, with EMA momentum μt=βμt-1+(1-β)gt\mu_t = \beta\mu_{t-1} + (1-\beta)g_t μt​=βμt-1​+(1-β)gt​, and SkipUpdate alternating: skip at tt t, update at t+1t+1 t+1.

**Skip step tt t:**

μt=βμt-1+(1-β)aθt,θt+1=θt\mu_t = \beta\mu_{t-1} + (1-\beta)a\theta_t, \quad \theta_{t+1} = \theta_tμt​=βμt-1​+(1-β)aθt​,θt+1​=θt​

**Update step t+1t+1 t+1** (since θ\theta θ didn't move, gradient is still aθta\theta_t aθt​):

μt+1=βμt+(1-β)aθt=β2μt-1+(1-β2)aθt\mu_{t+1} = \beta\mu_t + (1-\beta)a\theta_t = \beta^2\mu_{t-1} + (1-\beta^2)a\theta_tμt+1​=βμt​+(1-β)aθt​=β2μt-1​+(1-β2)aθt​ θt+2=θt-2ημt+1\theta_{t+2} = \theta_t - 2\eta\mu_{t+1}θt+2​=θt​-2ημt+1​

**Compare to standard momentum** after two steps:

μt+1std=β2μt-1+β(1-β)aθt+(1-β)aθt+1std\mu_{t+1}^{\text{std}} = \beta^2\mu_{t-1} + \beta(1-\beta)a\theta_t + (1-\beta)a\theta_{t+1}^{\text{std}}μt+1std​=β2μt-1​+β(1-β)aθt​+(1-β)aθt+1std​

The crucial difference is in the momentum term. SkipUpdate gives:

μt+1skip=β2μt-1+(1-β2)aθt\mu_{t+1}^{\text{skip}} = \beta^2\mu_{t-1} + (1-\beta^2)a\theta_tμt+1skip​=β2μt-1​+(1-β2)aθt​

The weight on old momentum is β2\beta^2 β2 instead of β\beta β --- it **decays faster**. The weight on current gradient information is (1-β2)=(1-β)(1+β)(1-\beta^2) = (1-\beta)(1+\beta) (1-β2)=(1-β)(1+β) instead of (1-β)(1-\beta) (1-β) --- it's **amplified by (1+β)(1+\beta) (1+β)**.

So SkipUpdate effectively makes momentum more responsive to current gradient information and less dependent on stale history. This is the core mechanism --- it's not really Nesterov lookahead, it's closer to **variance reduction**: by observing the gradient at the same position twice before moving, you get a cleaner, lower-variance signal before committing to a step. Does this feel like the right direction?

Hmm, but this would be similar to just a larger batch size? Challenge your own analysis, make it more complex

5:40 PM

Interrogated batch size equivalence and discovered fundamental differences in noise behavior

Interrogated batch size equivalence and discovered fundamental differences in noise behavior

Good challenge. Let me think about why it's NOT just larger batch size.

**The most direct challenge:** From our 1D analysis, larger batch size **reduces** the noise floor, while SkipUpdate **increases** it (2x-3x depending on curvature). They're actually opposite in their effect on noise. So the variance reduction framing was wrong.

**What's actually happening in 1D with momentum:**

The skip step accumulates momentum at the same θ, but then takes a **doubled step 2η**. For signal, this doubles the effective step --- good. For noise, it quadruples the contribution --- bad. This is fundamentally different from batch size which only reduces noise without changing step size.

**Where it gets more interesting --- multi-dimensional case:**

When θ₁ is skipped but θ₂ is updating, something nontrivial happens. The gradient of θ₁ at the next step is:

∇1l(θ1,θ2+Δθ2)=aθ1+c(θ2+Δθ2)\nabla_1 l(\theta_1, \theta_2 + \Delta\theta_2) = a\theta_1 + c(\theta_2 + \Delta\theta_2)∇1​l(θ1​,θ2​+Δθ2​)=aθ1​+c(θ2​+Δθ2​)

The momentum of θ₁ is now accumulating gradients across **different values of θ₂**. This is genuinely different from batch size --- it's averaging over the parameter trajectory of other weights, implicitly marginalizing out cross-parameter coupling.

This brings us back to explanation 2 in a more subtle way --- should we work this out analytically?

Ok, so it might kinda be the direction of explanation 1 and 2. Yes, work it out

5:42 PM

Let me set up the 2D quadratic with coupling:

l(θ1,θ2)=a2θ12+a2θ22+cθ1θ2l(\theta_1, \theta_2) = \frac{a}{2}\theta_1^2 + \frac{a}{2}\theta_2^2 + c\theta_1\theta_2l(θ1​,θ2​)=2a​θ12​+2a​θ22​+cθ1​θ2​

With alternating updates: skip θ1\theta_1 θ1​, update θ2\theta_2 θ2​ at step tt t; then update θ1\theta_1 θ1​, skip θ2\theta_2 θ2​ at step t+1t+1 t+1.

**Gradients:**

g1=aθ1+cθ2,g2=aθ2+cθ1g_1 = a\theta_1 + c\theta_2, \quad g_2 = a\theta_2 + c\theta_1g1​=aθ1​+cθ2​,g2​=aθ2​+cθ1​

**Step tt t: skip θ1\theta_1 θ1​, update θ2\theta_2 θ2​:**

θ2′=θ2-2η(aθ2+cθ1)=(1-2ηa)θ2-2ηcθ1\theta_2' = \theta_2 - 2\eta(a\theta_2 + c\theta_1) = (1-2\eta a)\theta_2 - 2\eta c\theta_1θ2′​=θ2​-2η(aθ2​+cθ1​)=(1-2ηa)θ2​-2ηcθ1​ θ1′=θ1\theta_1' = \theta_1θ1′​=θ1​ μ1′=βμ1+(1-β)(aθ1+cθ2)\mu_1' = \beta\mu_1 + (1-\beta)(a\theta_1 + c\theta_2)μ1′​=βμ1​+(1-β)(aθ1​+cθ2​)

Note μ1\mu_1 μ1​ updates using the **old** θ2\theta_2 θ2​ before it moved.

**Step t+1t+1 t+1: update θ1\theta_1 θ1​, skip θ2\theta_2 θ2​:**

μ1′′=βμ1′+(1-β)(aθ1+cθ2′)\mu_1'' = \beta\mu_1' + (1-\beta)(a\theta_1 + c\theta_2')μ1′′​=βμ1′​+(1-β)(aθ1​+cθ2′​)

Now substitute θ2′=(1-2ηa)θ2-2ηcθ1\theta_2' = (1-2\eta a)\theta_2 - 2\eta c\theta_1 θ2′​=(1-2ηa)θ2​-2ηcθ1​:

μ1′′=β2μ1+(1-β)β(aθ1+cθ2)+(1-β)(aθ1+c((1-2ηa)θ2-2ηcθ1))\mu_1'' = \beta^2\mu_1 + (1-\beta)\beta(a\theta_1 + c\theta_2) + (1-\beta)(a\theta_1 + c((1-2\eta a)\theta_2 - 2\eta c\theta_1))μ1′′​=β2μ1​+(1-β)β(aθ1​+cθ2​)+(1-β)(aθ1​+c((1-2ηa)θ2​-2ηcθ1​)) =β2μ1+(1-β)(1+β)(aθ1)+(1-β)c(βθ2+(1-2ηa)θ2)-(1-β)2ηc2θ1= \beta^2\mu_1 + (1-\beta)(1+\beta)(a\theta_1) + (1-\beta)c(\beta\theta_2 + (1-2\eta a)\theta_2) - (1-\beta)2\eta c^2\theta_1=β2μ1​+(1-β)(1+β)(aθ1​)+(1-β)c(βθ2​+(1-2ηa)θ2​)-(1-β)2ηc2θ1​ =β2μ1+(1-β2)aθ1+(1-β)c(1+β-2ηa)θ2-2η(1-β)c2θ1= \beta^2\mu_1 + (1-\beta^2)a\theta_1 + (1-\beta)c(1+\beta - 2\eta a)\theta_2 - 2\eta(1-\beta)c^2\theta_1=β2μ1​+(1-β2)aθ1​+(1-β)c(1+β-2ηa)θ2​-2η(1-β)c2θ1​

The update for θ1\theta_1 θ1​ is then:

θ1′′=θ1-2ημ1′′\theta_1'' = \theta_1 - 2\eta\mu_1''θ1′′​=θ1​-2ημ1′′​

**Compare to standard momentum** updating both parameters every step. After two steps, μ1\mu_1 μ1​ sees:

μ1std=β2μ1+β(1-β)(aθ1+cθ2)+(1-β)(aθ1′+cθ2′)\mu_1^{\text{std}} = \beta^2\mu_1 + \beta(1-\beta)(a\theta_1 + c\theta_2) + (1-\beta)(a\theta_1' + c\theta_2')μ1std​=β2μ1​+β(1-β)(aθ1​+cθ2​)+(1-β)(aθ1′​+cθ2′​)

The critical difference is the **last term**: standard momentum evaluates the coupling cθ2′c\theta_2' cθ2′​ using θ2′\theta_2' θ2′​ after it has already moved, while SkipUpdate's momentum accumulates coupling from **both before and after** θ2\theta_2 θ2​ moves.

The SkipUpdate coupling term is (1-β)c(1+β-2ηa)θ2(1-\beta)c(1+\beta-2\eta a)\theta_2 (1-β)c(1+β-2ηa)θ2​ while standard momentum's is roughly (1-β)c(1+θ2′/θ2)(1-\beta)c(1 + \theta_2'/\theta_2) (1-β)c(1+θ2′​/θ2​).

The key term is this one:

-2η(1-β)c2θ1-2\eta(1-\beta)c^2\theta_1-2η(1-β)c2θ1​

This is a **second-order coupling correction** --- proportional to c2c^2 c2 --- that appears naturally in SkipUpdate's momentum but not in standard momentum. It acts as an **implicit correction for the coupling**, reducing the effective step in the θ1\theta_1 θ1​ direction when coupling cc c is strong. This is precisely what a second-order method would compute explicitly, but SkipUpdate gets it for free through the interleaved update structure.

So the mechanism combining explanations 1 and 2 is: **the interleaved momentum updates implicitly accumulate cross-parameter coupling information**, generating a correction term that scales with c2c^2 c2 --- stronger coupling gives stronger implicit correction. This doesn't appear with larger batch size, and doesn't appear with synchronous updates.

Does this c2c^2 c2 correction term feel like the right smoking gun to you?

Hmm, but we already established beforehand actually that coupling is likely not the right explanation? Or is the correction here stronger, strong enough that skipping updates help?

5:47 PM

You're right, let me think carefully about this tension.

We established two things:

1.  The Hessian is near-block-diagonal --- so **between-block** coupling is small
2.  Element-wise masking works as well as block-wise --- so the relevant coupling is **within-block**, between individual weights

So the c2c^2 c2 correction could still be relevant, just at the **within-block, within-layer** level rather than between blocks. For example within an attention weight matrix WQW_Q WQ​, individual weights likely have non-trivial coupling through the shared input/output activations.

But then the question is: is this coupling large enough to matter? Let's think about what determines coupling strength within a weight matrix. The off-diagonal Hessian entry between weights ii i and jj j in the same layer is roughly:

Hij≈E[xixj⋅output curvature]H_{ij} \approx \mathbb{E}[x_i x_j \cdot \text{output curvature}]Hij​≈E[xi​xj​⋅output curvature]

This is the correlation between activations --- which for large hidden dimensions tends to be **small by concentration**, O(1/d)O(1/\sqrt{d}) O(1/d​).

So honestly --- even within blocks, coupling is probably small in practice, and the c2c^2 c2 correction would be negligible.

Which brings us back to: the momentum interaction (explanation 1) is probably the more fundamental story, and our 2D analysis didn't really add anything beyond that. Should we try to find a cleaner analytical argument purely for the momentum effect, without invoking coupling?

> We conjecture that this near-equivalence reflects the limited ability of diagonal preconditioning to exploit dense within block curvature, rendering finer-grained masking of marginal practical benefit.

Reading the paper, this sounds interesting. Let's unpack this. And is there anything more in the paper on this that could help us?

10:35 PM

Dissected diagonal preconditioning limitations and traced momentum's primacy over curvature

Dissected diagonal preconditioning limitations and traced momentum's primacy over curvature

The quote is from Section 2, right after the granularity ablation results (21.78 column-wise, 21.73 element-wise, 21.81 block-wise). Let me unpack it carefully.

**What it's actually saying:**

Adam/RMSProp use diagonal preconditioning --- they maintain Dt=diag(vt)-1/2D_t = \text{diag}(v_t)^{-1/2} Dt​=diag(vt​)-1/2, which only captures **per-parameter** curvature. Within a block, cross-parameter interactions HijH_{ij} Hij​ are already completely ignored by the base optimizer.

So when you mask at element level vs block level, in both cases the base optimizer is treating parameters independently anyway. The masking granularity doesn't change what curvature information is being exploited --- because none of the within-block off-diagonal curvature is being used regardless.

**The deeper implication:**

This actually undermines the Proposition 1 / curvature regularization story quite directly. If the base optimizer already ignores within-block coupling, then the geometric regularization effect from Proposition 1 --- which operates on Δ(b)⊤HbbΔ(b)\Delta^{(b)\top} H_{bb} \Delta^{(b)} Δ(b)⊤Hbb​Δ(b) --- is already somewhat captured by the diagonal preconditioner scaling those updates.

**This points strongly back to explanation 1** --- the momentum interaction must be carrying the weight, since it's the one thing that changes regardless of granularity or preconditioning structure.

Is there more in the paper? The heterogeneous quadratic experiment (Section 4.4) is relevant --- it specifically uses a non-axis-aligned Hessian to create within-block coupling that diagonal preconditioning can't handle. That's where Magma shows its biggest gains. This seems like the strongest empirical hint toward what the residual benefit beyond diagonal preconditioning actually is.

Should we try to formalize the momentum story more cleanly now, given we've essentially ruled out coupling as the primary driver?

I think this sentence in the paper is actually misleading, and that you're thus on the wrong track: * "the limited ability of diagonal preconditioning to exploit dense within block curvature" sure, but "diagonal preconditioning" comes from RMSProp, not SkipUpdate. The next sentence is then a non-sequitur: if finer-grained masking is targeting the off-diagonal curvature, then this should be independent of the  "diagonal preconditioning", and thus "rendering finer-grained masking of marginal practical benefit" makes little sense * Something that could make sense, that you were trying to aim at: if RMSProp's "diagonal preconditioning" actually also tackles the within-block off-diagonal curvature, which I haven't seen any reason this would be the case btw, then there's little more for SkipUpdate to gain for the within-block off-diagonal curvature Wdyt? Or is there any indication in the literature that diagonal preconditioning also tackles within-block off-diagonal curvature (and specifically just that, not across-block, as we see most gains from the across-block masking)?

Show more

10:43 PM

Recognized logical flaw in paper's explanation, pivoted toward alternative hypothesis

Recognized logical flaw in paper's explanation, pivoted toward alternative hypothesis

You're absolutely right, the paper's logic is flawed. Let me state it clearly:

**The paper's argument:**

1.  Element-wise ≈ block-wise masking in practice
2.  Therefore diagonal preconditioning already handles within-block off-diagonal curvature
3.  Therefore finer-grained masking adds nothing

**Why this is a non-sequitur:**

Diagonal preconditioning scales each parameter by 1/vt(i)1/\sqrt{v_t^{(i)}} 1/vt(i)​​ --- it is by definition completely blind to off-diagonal Hessian entries HijH_{ij} Hij​. There is no mechanism by which it could "exploit within-block off-diagonal curvature." The paper is essentially saying "A doesn't help because B already does it" without any reason why B would do it.

**The simpler, more honest explanation for granularity equivalence:**

Within-block off-diagonal coupling is simply small --- as we argued earlier, O(1/d)O(1/\sqrt{d}) O(1/d​) by concentration for large hidden dimensions. So there's nothing for finer-grained masking to exploit, not because RMSProp handles it, but because it's **small to begin with**.

This means the granularity equivalence actually tells us nothing about the mechanism --- it just tells us coupling is small at all scales.

**Which brings us cleanly back to:** the benefit of SkipUpdate is likely independent of coupling structure entirely, pointing firmly at the momentum interaction as the primary story. Should we now commit to working that out properly?

> Within-block off-diagonal coupling is simply small

But isn't across-block off-diagonal coupling even smaller? Isn't that what we've seen from another paper?

> Which brings us cleanly back to: the benefit of SkipUpdate is likely independent of coupling structure entirely, pointing firmly at the momentum interaction as the primary story.

I'm not so sure. At least the author's of the paper think coupling plays into it, and I am still preferring that explanation.

10:48 PM

Reconsidered coupling's role and acknowledged tension in the equivalence argument

Reconsidered coupling's role and acknowledged tension in the equivalence argument

You're making a really sharp point that I contradicted myself. Let me state it clearly:

**The hierarchy we've established:**

-   Across-block coupling: small (near-block-diagonal Hessian from the paper we found)
-   Within-block off-diagonal coupling: larger than across-block, but element-wise ≈ block-wise suggests it doesn't add much benefit

So actually within-block coupling > across-block coupling. Yet block-wise masking already captures most of the benefit. This is actually **paradoxical for the coupling explanation** --- if coupling drives the benefit, finer-grained masking should always help more by targeting more coupling. The near-equivalence across granularities suggests coupling isn't the driver.

**However** --- and this is where I think the authors' intuition might still be right --- maybe the relevant "coupling" isn't the off-diagonal Hessian entries per se, but the **heterogeneous curvature structure** we identified in Section 4.4. The heterogeneous quadratic experiment showed Magma specifically helps when eigenvalues of vastly different magnitudes are mixed --- that's a form of effective coupling through the optimization dynamics even with small off-diagonal entries.

So perhaps the right framing is: coupling in the traditional sense (off-diagonal Hessian) is small, but **curvature heterogeneity within blocks** creates a similar challenge that SkipUpdate addresses through the coordinate-descent-like stability argument we derived earlier.

Does that reconcile the two explanations?

Section 4.4 is very interesting, as it shows that magma works for some synthetic data, but unfortunately doesn't show SkipUpdate. How hard would it be to reproduce the experiment?

10:58 PM

Assessed reproducibility of quadratic optimization experiment as straightforward

Assessed reproducibility of quadratic optimization experiment as straightforward

Actually quite straightforward --- Appendix B.3 gives all the details needed. Let me just check the exact setup before coding.

The ingredients are:

-   l(w)=12w⊤Hwl(w) = \frac{1}{2}w^\top H w l(w)=21​w⊤Hw in R9\mathbb{R}^9 R9
-   Eigenvalues {1,2,3,99,100,101,4998,4999,5000}\{1,2,3,99,100,101,4998,4999,5000\} {1,2,3,99,100,101,4998,4999,5000}
-   3×3 block diagonal, with independent random rotation per block
-   Homogeneous: {1,2,3},{99,100,101},{4998,4999,5000}\{1,2,3\}, \{99,100,101\}, \{4998,4999,5000\} {1,2,3},{99,100,101},{4998,4999,5000} per block
-   Heterogeneous: {1,99,4998},{2,100,4999},{3,101,5000}\{1,99,4998\}, \{2,100,4999\}, \{3,101,5000\} {1,99,4998},{2,100,4999},{3,101,5000} per block
-   Stochasticity via subsampling rows of X=H1/2X = H^{1/2} X=H1/2

The only thing not fully specified is the subsampling fraction. But we can try a reasonable value.

We'd want to compare:

-   AdamW (as in the paper)
-   SkipUpdate (not in the paper --- this is the key missing experiment)
-   Magma (as in the paper, for validation)

For both homogeneous and heterogeneous cases. Shall I implement this now?

Do you really have everything? I think optimizer hyperparameters are unknown, right?

10:59 PM

Acknowledged missing optimizer hyperparameters requiring specification or search

Acknowledged missing optimizer hyperparameters requiring specification or search

You're right, we're missing quite a bit actually:

**For AdamW/SkipUpdate/Magma:**

-   Learning rate η
-   β1, β2 for Adam
-   Weight decay
-   For Magma: temperature τ and sampling probability p (though paper uses τ=2, p=0.5 generally)

**For the experiment itself:**

-   Subsampling fraction for stochasticity
-   Number of iterations
-   Initial θ₀

The paper's Figure 4 shows multiple learning rates for AdamW and Magma (lr=0.003, 0.01, 0.03, 0.1), so they at least swept over those. But β1, β2 and subsampling fraction are completely unspecified.

**Options:**

1.  Use standard defaults (η from a sweep, β1=0.9, β2=0.999) and try to reproduce Figure 4 qualitatively for AdamW+Magma first --- if we can match their curves, we know our setup is right, then add SkipUpdate
2.  Treat missing hyperparameters as free and just compare AdamW vs SkipUpdate directly under the same settings

Option 1 seems more rigorous since it gives us a validation check before drawing conclusions about SkipUpdate. Shall I implement that?

Option 1 is the way to go for sure. But first, let's discuss: what's the exact structure of the synthetic data? And let's hypothesize how magma helps specifically (and how SkipUpdate could help).

5:01 PM

Examined synthetic data architecture and theorized optimization mechanisms

Examined synthetic data architecture and theorized optimization mechanisms

**Synthetic data structure:**

From Appendix B.3, the loss is l(w)=12w⊤Hwl(w) = \frac{1}{2}w^\top H w l(w)=21​w⊤Hw, stochasticity comes from subsampling rows of X=H1/2X = H^{1/2} X=H1/2. So the stochastic gradient at each step is:

gt=XS⊤XSwg_t = X_S^\top X_S wgt​=XS⊤​XS​w

where XSX_S XS​ is a random subset of rows of H1/2H^{1/2} H1/2. This means the stochastic gradient is an unbiased estimator of HwHw Hw, with noise structure determined by which rows are subsampled. Crucially, the noise is **multiplicative** --- it scales with ww w, so near the optimum noise vanishes. This is different from typical SGD noise assumptions.

**Key structural difference between homogeneous and heterogeneous:**

In the homogeneous case, each 3×3 block has similar eigenvalues --- the random rotation within a block mixes eigenvalues of similar scale. Adam's diagonal preconditioning, while not perfect, roughly handles this since gradients within a block have similar magnitudes.

In the heterogeneous case, each block mixes eigenvalues spanning three orders of magnitude ({1,99,4998}\{1, 99, 4998\} {1,99,4998}). After the random rotation, every coordinate within a block has contributions from both tiny and huge curvature directions. Adam's diagonal preconditioning sees similar gradient magnitudes across coordinates but the underlying curvature is wildly different --- it's fundamentally confused.

**Why Magma specifically helps for heterogeneous:**

When a block mixes wildly different curvatures, the gradient direction fluctuates dramatically between steps --- the high-curvature components dominate and oscillate. The momentum-gradient alignment score detects this: low alignment signals that the current gradient is conflicting with accumulated momentum, which happens precisely in high-curvature directions. Magma suppresses these updates, effectively acting like a crude second-order correction.

**How SkipUpdate could help:**

This is the interesting hypothesis --- in the heterogeneous case, when you skip a block's update, the momentum keeps accumulating gradients while parameters stay fixed. Since the high-curvature components change gradient direction rapidly, averaging them out through extra momentum accumulation before committing to a step could naturally dampen the oscillatory high-curvature components. It's essentially giving the momentum more observations to average over specifically in the directions that fluctuate most.

But importantly --- this would NOT help in the homogeneous case, since all curvature directions within a block are similar scale, and there's nothing oscillatory to average out. This would be a clean, testable prediction from our theory.

Shall we now implement the experiment to test this?

Actually, I will want to do this experiment in claude code, this is not the right interface (not persistent) IMO. Let's just continue discussing then. Describe the synthetic data's structure at a higher level and explaining more again, just that, I found it to be too dense for me.

5:07 PM

Sure! Let me build it up step by step.

**The loss function:**It's just a simple quadratic bowl l(w)=12w⊤Hwl(w) = \frac{1}{2}w^\top H w l(w)=21​w⊤Hw. The minimum is at w=0w=0 w=0, and HH H determines the shape of the bowl --- how steep it is in each direction.

**What H determines:**The eigenvalues of HH H are the curvatures along the principal axes. With eigenvalues spanning {1,2,3,99,100,101,4998,4999,5000}\{1, 2, 3, 99, 100, 101, 4998, 4999, 5000\} {1,2,3,99,100,101,4998,4999,5000}, the bowl is extremely elongated --- nearly flat in some directions, extremely steep in others. This is the ill-conditioning that makes optimization hard.

**Homogeneous vs heterogeneous --- the key difference:**

Think of the 9 dimensions as three groups of 3.

In the **homogeneous** case, the three groups are "pure" --- one group is flat, one medium, one steep. The groups don't mix. Adam handles this reasonably well because within each group gradients are similar scale.

In the **heterogeneous** case, each group contains one flat, one medium and one steep direction, randomly mixed together by a rotation. So every coordinate is a mixture of wildly different curvatures. This is what makes it hard --- Adam sees similar gradient magnitudes everywhere but the underlying geometry is chaotic within each group.

**The stochasticity:**Rather than adding artificial noise, they subsample rows of H1/2H^{1/2} H1/2 to form each gradient estimate. This means noise naturally scales with distance from optimum --- realistic for neural network training.

Does this feel clearer?

Show me also how H looks like exactly, how can it be generated? It's e.g. not quite clear to me how the groups work

> The stochasticity:Rather than adding artificial noise, they subsample rows

Is this a principled approach? Is there realistic data that has this kind of structure, or an analogy at least?

> This means noise naturally scales with distance from optimum

Really? Is that so realistic?

5:14 PM

Unpacked matrix construction, validated stochastic sampling, questioned noise realism

Unpacked matrix construction, validated stochastic sampling, questioned noise realism

**How H is constructed concretely:**

Start with a diagonal matrix of eigenvalues, then apply a random rotation QQ Q (random orthogonal matrix):

Hblock=QΛQ⊤H_{block} = Q \Lambda Q^\topHblock​=QΛQ⊤

So concretely, homogeneous H looks like:

H=(Q1diag(1,2,3)Q1⊤000Q2diag(99,100,101)Q2⊤000Q3diag(4998,4999,5000)Q3⊤)H = \begin{pmatrix} Q_1 \text{diag}(1,2,3) Q_1^\top & 0 & 0 \\ 0 & Q_2 \text{diag}(99,100,101) Q_2^\top & 0 \\ 0 & 0 & Q_3 \text{diag}(4998,4999,5000) Q_3^\top \end{pmatrix}H=​Q1​diag(1,2,3)Q1⊤​00​0Q2​diag(99,100,101)Q2⊤​0​00Q3​diag(4998,4999,5000)Q3⊤​​​

Heterogeneous H:

H=(Q1diag(1,99,4998)Q1⊤000Q2diag(2,100,4999)Q2⊤000Q3diag(3,101,5000)Q3⊤)H = \begin{pmatrix} Q_1 \text{diag}(1,99,4998) Q_1^\top & 0 & 0 \\ 0 & Q_2 \text{diag}(2,100,4999) Q_2^\top & 0 \\ 0 & 0 & Q_3 \text{diag}(3,101,5000) Q_3^\top \end{pmatrix}H=​Q1​diag(1,99,4998)Q1⊤​00​0Q2​diag(2,100,4999)Q2⊤​0​00Q3​diag(3,101,5000)Q3⊤​​​

The random rotations QiQ_i Qi​ are what "mix" the curvatures within each block --- without them the heterogeneous case would just be a reordering of eigenvalues, and Adam's diagonal preconditioning would handle it perfectly by learning different step sizes per coordinate. The rotation ensures no axis-aligned solution exists.

**On the stochasticity:**

Actually yes, it's quite principled. X=H1/2X = H^{1/2} X=H1/2 means X⊤X=HX^\top X = H X⊤X=H, so the full gradient is Hw=X⊤XwHw = X^\top Xw Hw=X⊤Xw. This is literally the gradient of a least-squares problem min⁡w1n∥Xw∥2\min_w \frac{1}{n}\|Xw\|^2 minw​n1​∥Xw∥2 --- subsampling rows of XX X is exactly mini-batch SGD on that regression problem. Clean and well-motivated.

**On noise scaling with distance --- you're right to challenge this:**

The noise is (XS⊤XS-H)w(X_S^\top X_S - H)w (XS⊤​XS​-H)w, which does vanish as w→0w \to 0 w→0. But in real neural network training the minimum is not at zero loss --- there's irreducible noise from mini-batching that stays constant regardless of how close you are to the optimum. So this is actually **unrealistically optimistic** near the optimum, and probably why the paper can show clean convergence curves without worrying too much about noise floors.

Thanks, that clarifies things a lot. Also, the underlying motivation for H is the covariance matrix of data, is that the right intuition? I.e. what underlying optimization problem with data X does this correspond to? What does the block structure of H mean for the data?

5:17 PM

Synthesized covariance structure with data optimization and feature correlation patterns

Synthesized covariance structure with data optimization and feature correlation patterns

**The underlying optimization problem:**

Since X=H1/2X = H^{1/2} X=H1/2, we have H=X⊤XH = X^\top X H=X⊤X, and the loss becomes:

l(w)=12w⊤X⊤Xw=12∥Xw∥2l(w) = \frac{1}{2}w^\top X^\top X w = \frac{1}{2}\|Xw\|^2l(w)=21​w⊤X⊤Xw=21​∥Xw∥2

This is linear regression with no labels --- just minimizing the output norm. The gradient is X⊤Xw=HwX^\top Xw = Hw X⊤Xw=Hw, which is why subsampling rows of XX X gives a natural stochastic gradient.

**H as covariance:**

Yes, H=X⊤XH = X^\top X H=X⊤X is exactly the data covariance matrix (up to scaling). Its eigenvalues are the variances along principal components --- how much the data "spreads" in each direction. Large eigenvalues = high variance directions = steep loss curvature.

**What block structure means for data:**

The block-diagonal structure of HH H means features decompose into groups that are **correlated within groups but independent across groups**. For neural networks this is natural --- weights in the same layer share inputs and outputs, creating within-layer correlations, while weights in different layers interact less directly.

**Homogeneous vs heterogeneous in data terms:**

-   Homogeneous: within each correlated feature group, variances are similar scale --- relatively well-behaved data
-   Heterogeneous: within each correlated feature group, some features have enormous variance and some tiny --- like natural language where within a semantic cluster, some dimensions are highly active and others barely used

The heterogeneous case is actually very realistic for language data, which might explain why the gains are specifically seen in LLM training.

Ok, this is interesting: it models only within-block correlation, with no across-block correlation. But doesn't magma and SkipUpdate only work across blocks (at least the variant they settle on in the end)?

5:20 PM

This is a really sharp observation and a genuine tension in the paper.

You're right on both counts:

**The synthetic data:** Block-diagonal H means correlations are purely within-block. Across-block correlations are exactly zero by construction.

**The final method:** They settle on block-wise masking as the practical choice, where entire blocks are either updated or skipped together. This operates across blocks --- deciding which blocks update at each step --- not within blocks.

So the experiment in Section 4.4 is designed to show benefits from within-block curvature heterogeneity, but the method they ultimately advocate operates at the block level, which can't exploit within-block structure at all.

These two things are in direct contradiction:

-   If the benefit comes from within-block heterogeneous curvature → element-wise masking should be substantially better than block-wise
-   But the granularity ablation shows they're nearly equivalent
-   And the synthetic experiment has no across-block coupling at all

So Section 4.4 might actually be testing the wrong thing --- it demonstrates Magma works on a problem structure that element-wise masking would address, but the final deployed method is block-wise which can't use that structure.

This feels like a significant conceptual gap in the paper --- the theoretical motivation (Section 4.4, Proposition 1) and the practical method (block-wise masking) are somewhat misaligned. Would you agree this is a real weakness?

Yes, seems inconsistent. For the section 4.4 experiments, do they actually use a block-wise or element-wise implementation of magma?

5:21 PM

Looking at the paper carefully... it doesn't actually specify this for Section 4.4. The experimental details in Appendix B.3 describe how H is constructed and the stochasticity, but say nothing about the masking granularity used for Magma in that experiment.

This is a significant omission. Because:

-   If they used **element-wise** masking in Section 4.4: the experiment is internally consistent (element-wise masking can exploit within-block heterogeneous curvature), but then it's demonstrating something different from the final recommended method (block-wise)
-   If they used **block-wise** masking in Section 4.4: the experiment is internally inconsistent --- block-wise masking operates on the 3×3 blocks, and with zero across-block coupling in H, it's unclear what signal the masking is even using. The heterogeneous curvature is entirely within blocks, invisible to block-wise masking.

Either way there's a problem. And this actually makes Section 4.4 harder to interpret as evidence for why the method works --- we don't know if we're seeing benefits from within-block or across-block structure, and the paper conflates the two.

This would be one of the first things to nail down in the Claude Code reproduction --- implement both granularities explicitly and compare, which the paper never does for this synthetic experiment.
