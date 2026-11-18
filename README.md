# Robot Modeling

새 A0509 오프라인 물리 잔차 실험의 코드, 재현 명령, 실제 결과는
[src/model/README.md](src/model/README.md)에 정리했습니다.

Doosan 로봇 모델링 작업 공간입니다. 현재 A0509 학습은 기존 프로젝트 `D:\Code\sdml-doosan-robot`의 데이터와 MuJoCo 장면을 **D 드라이브에서 직접 읽습니다.** 전체 `data` 폴더 복사는 필요하지 않습니다.

## 폴더

- `src/model/`: 새 모델 코드
- `data/`: 원본에서 검증해 복사한 학습·검증·시험 데이터셋
- `result/`: 이 작업 공간에서 새로 생성할 결과
- `.venv/`: `uv`로 만든 Python 3.12 가상환경

기존 프로젝트의 개요는 `D:\Code\sdml-doosan-robot\README.md`, 모델 설명은 `D:\Code\sdml-doosan-robot\model\README.md`를 참고하세요.

## 로컬 학습 데이터

원본의 학습용 데이터셋 6개, 총 60개 파일(약 271 MiB)을 확인하고 복사했습니다. 원본과 복사본의 SHA-256이 모두 일치합니다.

| 경로 | 검증 결과 |
| --- | --- |
| `data/wide16_v3_joint_dataset/` | 7조건 × 16궤적, 112개 trial, 202,608행 |
| `data/grid16_legacy_joint_dataset_m0_m4/` | 7조건 × 16궤적, 112개 trial, 93,912행 |
| `data/grid16_fourier_pkan_csv/` | 7조건의 학습·검증·시험·진단 CSV |
| `data/grid16_pkan_csv/` | 이전 5조건의 학습·검증·시험·진단 CSV |
| `data/residual_data/rbd_v1/` | 13,760행 |
| `data/residual_data/dynamics10_v1/` | 4,630행 |

`data/residual_data/`의 원본은 기존 프로젝트의 `model/residual_data/`입니다. ROS bag(`*.db3`), 반복 trial의 원시 기록, 그림, 영상, 체크포인트는 복사하지 않았습니다. 토크 라벨은 MuJoCo 역동역학 값이며 실측 모터 토크가 아닙니다. 현재 `src/model` 실험은 원본 D 드라이브를 직접 읽는 설정입니다.

학습 데이터만 다시 복사하고 원본·복사본 해시를 확인하려면 PowerShell에서 실행하세요. 기존 파일의 해시가 같으면 건너뛰며, 다른 파일을 삭제하지 않습니다.

```powershell
.\scripts\copy_training_data.ps1
```

Git 저장소는 [sepengsu/pkan-robot-modeling](https://github.com/sepengsu/pkan-robot-modeling)입니다. 대용량 `data/`, 새 실행 결과 `result/`, `.venv/`는 업로드 대상에서 제외하고 폴더 자리표시자만 버전 관리합니다.

## 학습 환경

검증된 PyTorch 환경은 기존 프로젝트의 `model/.venv`입니다. PowerShell에서 이 작업 공간 루트로 이동한 후:

```powershell
$py = 'D:\Code\sdml-doosan-robot\model\.venv\Scripts\python.exe'
& $py -m src.model prepare
& $py -m src.model train
& $py -m src.model evaluate
```

입력은 D 드라이브에서 읽고, 준비된 소형 데이터·체크포인트·평가 결과는 `result/structured_kan/`에 저장합니다.
