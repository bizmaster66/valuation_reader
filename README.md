# 등기사항전부증명서 주식변동 추출기

Google Drive 폴더에 있는 등기사항전부증명서(PDF)를 읽어,
주식 변동 이력을 정리한 Google Spreadsheet를 생성합니다.

## 기능
- Drive 폴더 ID 입력 → PDF 자동 처리
- 발행주식/자본금/변경연월일/등기연월일 등 추출
- 우선주 합산 검증 실패 시 해당 셀 빨간색 표시
- 기업별 구분선(굵은 테두리) 표시
- 이미 처리된 파일은 "읽음" 상태로 표시, 재분석 가능
- 결과는 `result` 폴더에 스프레드시트로 저장

## 실행
```bash
streamlit run app.py
```

## Streamlit Cloud 배포
1. GitHub에 푸시
2. Streamlit Cloud에서 리포지토리 연결
3. Secrets 설정

### Secrets 예시
Streamlit Cloud > App settings > Secrets 에 아래 형식으로 등록:
```toml
gcp_service_account = """
{ ... 서비스 계정 JSON ... }
"""
```

## 결과 파일명
`{폴더명}_result_YYYYMMDD`
