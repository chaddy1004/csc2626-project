# Montezuma's Revenge

The [Atari-HEAD dataset](Thttps://arxiv.org/pdf/1903.06754.pdf) provides high-quality human demonstrations of 20 Atari gameplays, including *Montezuma's Revenge* (MR). We treat the frame-by-frame data as expert demonstrations to teach an agent how to play MR in an imitation learning setting.

## Overview

| Trial      | Episode | Frames | Total processed reward |
| ---------- | ------- | ------ | ----- |
| 284 | 1 | 15650 | 35 |
| 285 | 1 | 10497 | 10 |
| 285 | 2 | 5995 | 8 |
| 287 | 1 | 16193 | 37 |
| 291 | 1 | 16720 | 40 |
| 324 | 1 | 15894 | 36 |
| 333 | 1 | 16686 | 39 |
| 340 | 1 | 17052 | 37 |
| 359 | 1 | 17095 | 38 |
| 365 | 1 | 17066 | 40 |
| 371 | 1 | 16546 | 37 |
| 385 | 1 | 16955 | 38 |
| 398 | 1 | 16882 | 39 |
| 402 | 1 | 17053 | 42 |
| 416 | 1 | 16991 | 44 |
| 429 | 1 | 16934 | 40 |
| 436 | 1 | 17065 | 41 |
| 459 | 1 | 17119 | 41 |
| 469 | 1 | 16757 | 42 |
| 480 | 1 | 17082 | 44 |
| 493 | 1 | 17003 | 42 |
| __Total__  | 21      | 335235 | 

The rewards in MR are generally sparse; a `sgn` function is used to process the original rewards and a reward of -100 is awarded to the frame (terminal state) ending each episode.