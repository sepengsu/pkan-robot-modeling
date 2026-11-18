# Robot Modeling

Doosan 로봇 모델링 작업 공간입니다. 기존 프로젝트 `D:\Code\sdml-doosan-robot`을 참고하며, 원본의 `data` 폴더 전체를 이 작업 공간의 `data`로 복사합니다.

## 폴더

- `src/model/`: 새 모델 코드
- `data/`: `D:\Code\sdml-doosan-robot\data`에서 가져온 원본 데이터와 기존 실험 산출물
- `result/`: 이 작업 공간에서 새로 생성할 결과
- `.venv/`: `uv`로 만든 Python 3.12 가상환경

기존 프로젝트의 개요는 `D:\Code\sdml-doosan-robot\README.md`, 모델 설명은 `D:\Code\sdml-doosan-robot\model\README.md`를 참고하세요. 이곳의 `data`는 원본에서 복사한 독립 사본이므로 이후 원본 변경 사항은 자동으로 반영되지 않습니다.

Git 저장소는 [sepengsu/pkan-robot-modeling](https://github.com/sepengsu/pkan-robot-modeling)입니다. 대용량 `data/`, 새 실행 결과 `result/`, `.venv/`는 업로드 대상에서 제외하고 폴더 자리표시자만 버전 관리합니다.

## 가상환경

PowerShell에서 작업 공간 루트로 이동한 후:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

필요한 패키지는 작업 목적에 따라 `uv pip install ...`로 설치하세요.
