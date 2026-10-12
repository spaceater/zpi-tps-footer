# @semanaut/zpi-tps-footer

Live model **output speed (tokens/s)** in the [pi](https://github.com/earendil-works/pi) footer, plus the session id.

The headline feature is the throughput readout: while the model is generating, every completed token is counted and the current output rate — `<t/s>` — is displayed after the model name, updating live. It is always rendered (`NULL t/s` only before the first measurement), so you can watch how fast your model actually streams.

Replaces pi's built-in footer via `ctx.ui.setFooter()` and reproduces it, adding exactly two elements:

- line 2, after the model name: `• <t/s>` — the live output rate
- line 1, right-aligned: the session id

## Install

```bash
pi install npm:@semanaut/zpi-tps-footer
```
