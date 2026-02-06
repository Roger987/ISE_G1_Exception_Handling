#!/bin/bash

echo "app_id,python_techs" > analysis_csv/python_frameworks.csv 
#echo "tech,count" > analysis_csv/stats.csv

python_count=0
flask_count=0
fastapi_count=0
django_count=0
tornado_count=0
other_count=0

for app_dir in vibe_dataset/apps/*/ ; do
  app_id=$(basename "$app_dir")
  tech=""
  
  # Django if contains manage.py
  if [ -f "$app_dir/manage.py" ]; then
    tech="Django"
  else
    # Check requirements.txt
    if [ -f "$app_dir/requirements.txt" ]; then
      candidate=""
      grep -qi flask "$app_dir/requirements.txt" >/dev/null 2>&1 && candidate="Flask"
      grep -qi "fastapi\|uvicorn" "$app_dir/requirements.txt" >/dev/null 2>&1 && candidate="FastAPI"
      grep -qi django "$app_dir/requirements.txt" >/dev/null 2>&1 && candidate="Django"
      
      # Verify candidate in the file .py
      if [ -n "$candidate" ]; then
        found=false
        for file in "$app_dir"/*.py; do
          [ -f "$file" ] || continue
          case "$candidate" in
            Flask) grep -qi "from flask import\|Flask(__name__)\|@app.route" "$file" >/dev/null 2>&1 && found=true && break ;;
            FastAPI) grep -qi "FastAPI()\|APIRouter\|@router\|uvicorn.*main:app\|BaseModel\|@app.get\|@app.post" "$file" >/dev/null 2>&1 && found=true && break ;;
            Django) grep -qi "django.http\|django.shortcuts\|django.views" "$file" >/dev/null 2>&1 && found=true && break ;;
          esac
        done
        [ "$found" = true ] && tech="$candidate"
      fi
    fi
    
    # Minor frameworks as Tornado
    if [ -z "$tech" ]; then
      for file in "$app_dir"/{app.py,main.py}; do
        [ -f "$file" ] || continue
        grep -qi "tornado.*application" "$file" >/dev/null 2>&1 && tech="Tornado" && break
        grep -qi "socket\|http.server" "$file" >/dev/null 2>&1 && tech="Other" && break
      done
    fi
  fi
  
  # Save results
  if [ -n "$tech" ]; then
    echo "$app_id,$tech" >> analysis_csv/python_frameworks.csv
    ((python_count++))
    case "$tech" in
      Flask) ((flask_count++)) ;;
      FastAPI) ((fastapi_count++)) ;;
      Django) ((django_count++)) ;;
      Tornado) ((tornado_count++)) ;;
      Other) ((other_count++)) ;;
    esac

    echo "$app_id,$tech"
  fi
done

# Write stats
#echo "Python apps,$python_count" >> analysis_csv/stats.csv
#echo "Flask,$flask_count" >> analysis_csv/stats.csv
#echo "FastAPI,$fastapi_count" >> analysis_csv/stats.csv
#echo "Django,$django_count" >> analysis_csv/stats.csv
#echo "Tornado,$tornado_count" >> analysis_csv/stats.csv
#echo "Other,$other_count" >> analysis_csv/stats.csv

echo "Scan completed. Results saved in analysis_csv/python_frameworks.csv"
echo "Python frameworks found: $python_count"
