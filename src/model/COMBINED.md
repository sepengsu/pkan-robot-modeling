# Legacy + wide16_v3 공동 M0–M5 실험

이 실험은 `data/grid16_legacy_joint_dataset_m0_m4/`와 `data/wide16_v3_joint_dataset/`를 함께 사용한다. 각 archive의 SHA-256, 7조건 × 16궤적의 112개 trial, 입력 형식, MuJoCo 장면 XML 해시를 준비 전에 검증한다. 장면 XML과 RBD 구현은 기존 `D:\Code\sdml-doosan-robot`에서 읽는다. ROS bag과 원시 trial CSV는 사용하지 않는다.

## 분할과 라벨

두 캠페인의 원래 분할은 서로 달라서 새 실험에서는 **궤적 ID 기준으로 동일하게 다시 분할**한다. 같은 궤적 ID가 두 캠페인의 학습과 검증·시험에 동시에 들어가지 않는다. Legacy의 기존 진단용 T1–T4도 새 실험에서는 학습에 포함한다.

| 분할 | 궤적 | Legacy | wide16_v3 | 합계 |
| --- | --- | ---: | ---: | ---: |
| 학습 | T1–T11, T15 | 73,514 | 151,956 | 225,470 |
| 검증 | T12 | 4,697 | 12,663 | 17,360 |
| 시험 | T13, T14, T16 | 15,701 | 37,989 | 53,690 |
| 전체 | T1–T16 | 93,912 | 202,608 | 296,520 |

두 archive 모두 `q, dq, ddq` 각 6축, bare/AG145/mDPG 원핫 3개, 페이로드 질량 1개로 구성된 22입력을 가진다. 조건은 bare 0 kg, AG145와 mDPG 각각 0/0.5/1.0 kg이다. 완전히 동일한 22입력 행은 캠페인 간 0개였다. 다만 AG145 세 조건의 장면 해시가 캠페인 간 달라서 각 캠페인의 저장된 장면으로 라벨을 별도 재계산한다.

- A: 저장된 MuJoCo `qfrc_inverse`. M1은 같은 장면의 RBD이므로 A의 오차 0은 구성상 일치 확인이다.
- B: 같은 `q,dq,ddq`에서 관절 `frictionloss`를 `[0.40, 0.35, 0.25, 0.15, 0.12, 0.08]` Nm으로 바꿔 MuJoCo `mj_inverse`로 재계산한 토크.
- M0는 저장된 0.8배 tool/payload 관성 명목 RBD, M1은 원래 장면의 정확한 RBD이다. 둘 다 학습 가중치가 없다.
- M2–M5는 `B − A`의 6축 잔차를 학습한다. M2·M4·M5는 `q,dq` 12입력, M3는 `dq` 6입력을 사용하며 condition과 campaign 코드는 학습 입력에 넣지 않는다.

wide16_v3의 표본 수가 더 많으므로 학습·조기 종료 손실에서는 두 캠페인의 **총 가중치를 동일하게** 둔다. 모든 행은 매 epoch 사용한다. 평가 파일에는 전체 및 캠페인별·조건별·궤적별 원래 행 수 기준 RMSE를 함께 기록한다. `combined_config.json`의 3개 시드, 최대 60 epoch, batch 1024, 조기 종료 설정을 사용했다.

## 실행

PowerShell에서 저장소 루트 기준. 새 작업 공간이라면 먼저 `uv`로 Python 3.12 환경과 의존성을 준비한다:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -r .\src\model\requirements-combined.txt
```

그다음 실행한다:

```powershell
.\scripts\run_combined_models.ps1 -Stage Validate
.\scripts\run_combined_models.ps1 -Stage All
```

`All`은 검증, 라벨 준비, M2–M5 학습, M0–M5 평가를 순서대로 실행한다. 기존 결과 폴더가 있으면 새 시각 이름의 폴더를 사용해 결과를 덮어쓰지 않는다. 개별 단계는 `-Stage Prepare`, `-Stage Train`, `-Stage Evaluate`와 동일한 `-OutputRoot`로 실행한다. Python 기본값은 이 저장소의 `.venv/Scripts/python.exe`이며 `-Python`으로 바꿀 수 있다. 출력은 `result/combined_legacy_wide_v3/` 아래에 저장하고 Git에는 올리지 않는다.

## 이번 실행의 시험 결과

T13/T14/T16, 7조건·6축의 전체 RMSE [Nm]. M2–M5의 `±`는 3개 시드 간 표본 표준편차다.

| 모델 | 전체 | Legacy | wide16_v3 |
| --- | ---: | ---: | ---: |
| M0 명목 RBD | 0.70567 | 0.51781 | 0.77004 |
| M1 정확한 장면 RBD | 0.21462 | 0.20661 | 0.21785 |
| M2 RBD + MLP | **0.01755 ± 0.00109** | 0.01780 ± 0.00090 | 0.01743 ± 0.00127 |
| M3 RBD + 관절별 spline | 0.02988 ± <0.00001 | 0.02996 ± <0.00001 | 0.02985 ± <0.00001 |
| M4 RBD + 결합 KAN | 0.03167 ± 0.01100 | 0.03529 ± 0.00789 | 0.02983 ± 0.01284 |
| M5 RBD + 단계별 구조화 KAN | 0.02750 ± 0.00446 | 0.02981 ± 0.00051 | 0.02630 ± 0.00651 |

이번 설정에서는 M2가 가장 낮은 전체 시험 오차를 냈다. M4·M5에는 시드 간 편차가 있다. 재현 가능한 시드별 수치는 `evaluation/metrics.csv`와 `metrics.json`, 학습 이력은 `models/seed_*/`에 있고, 그림은 `evaluation/figures/`에 있다.

이 결과는 **시뮬레이션에서 인위적으로 추가한 관절 마찰**을 예측한 것이다. 실측 모터 토크, 로봇의 실제 마찰 또는 안전한 실로봇 동작을 검증한 결과가 아니다. 원래 한 캠페인만 사용한 실험과 분할·배치 크기가 달라 수치를 직접적인 모델 개선율로 해석하면 안 된다.
