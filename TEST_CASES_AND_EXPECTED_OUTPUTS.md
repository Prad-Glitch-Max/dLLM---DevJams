# 🧪 DiffAgent Sample Test Cases & Expected Outputs

This document details the complete suite of verification test cases that can be executed in DiffAgent. It explains the underlying routing logic, early-commitment trigger steps, canonical tool spans, retrieved passages, and exact expected synthesized outputs for each query.

---

## 📋 Comprehensive Test Cases Table

| # | Test Case Query | Expected Tool | Canonical Tool Span | Early Trigger | Target Domain |
|---|---|---|---|---|---|
| **1** | `What is the weather in chennai?` | `weather` | `weather ( location = 'Chennai' )` | Step 7 / 10 | Live Open-Meteo API |
| **2** | `Will i need an umbrella in chennai today?` | `weather` | `weather ( location = 'Chennai' )` | Step 7 / 10 | Conversational Weather Advisory |
| **3** | `What are the library opening hours?` | `campus` | `campus ( query = 'library opening hours' )` | Step 7 / 10 | Campus RAG (`library.txt`) |
| **4** | `What are the hostel facilities available?` | `campus` | `campus ( query = 'hostel facilities available' )` | Step 7 / 10 | Campus RAG (`hostel.txt`) |
| **5** | `What is the attendance requirement?` | `campus` | `campus ( query = 'attendance requirement' )` | Step 7 / 10 | Campus RAG (`academic.txt`) |
| **6** | `What student services are available on the campus?` | `campus` | `campus ( query = 'student services on campus' )` | Step 7 / 10 | Campus RAG (`campus_services.txt`) |
| **7** | `calculate 125*48` | `calculator` | `calculator ( expression = '125*48' )` | Step 7 / 10 | Safe Python AST Evaluator |
| **8** | `if i study for 3 hours for 7 days, then how many hours is that?` | `calculator` | `calculator ( expression = '3 * 7' )` | Step 7 / 10 | Natural Language Math Reasoning |
| **9** | `What are the library hours and the hostel services available?` | `campus` | `campus ( query = 'library hours AND hostel services' )` | Step 7 / 10 | Multi-Domain Compound RAG |

---

## 🔍 Detailed Test Case Walkthroughs

### Test Case 1: General Weather Query
- **Query:** `What is the weather in chennai?`
- **Tool Routed:** `weather`
- **Canonical Tool Span:** `weather ( location = 'Chennai' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Live Tool Result:** Fetches live temperature, apparent temperature, humidity, and wind speed from Open-Meteo API.
- **Expected Final Response:**
```markdown
⛅ The current weather in **Chennai, India** is **31.4°C** with **Partly cloudy**. It feels like **35.3°C**, relative humidity is **67%**, and wind speed is **17.0 km/h**.
```

---

### Test Case 2: Conversational Umbrella / Rain Query
- **Query:** `Will i need an umbrella in chennai today?`
- **Tool Routed:** `weather`
- **Canonical Tool Span:** `weather ( location = 'Chennai' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Logic:** Evaluates live precipitation levels ($0.0\text{ mm}$) and WMO condition codes.
- **Expected Final Response (Dry Condition):**
```markdown
☀️ **No need to carry an umbrella.** The weather in **Chennai, India** today is **partly cloudy** with **0 mm** precipitation and a temperature of **31.4°C**.
```
*(If rain/drizzle is active, returns: `🌧️ Yes, please carry an umbrella! There is light rain (precipitation: X mm) in Chennai...`)*

---

### Test Case 3: Library Opening & Closing Hours
- **Query:** `What are the library opening hours?`
- **Tool Routed:** `campus`
- **Canonical Tool Span:** `campus ( query = 'What are the library opening hours' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Retrieved Document:** `library.txt` (Section: *1. Library Timings & Opening Hours*)
- **Expected Final Response:**
```markdown
**Library Timings & Opening Hours:**
• Monday to Friday: 8:00 AM to 10:00 PM (Opening time: 8:00 AM, Closing time: 10:00 PM)
• Saturday and Sunday: 9:00 AM to 8:00 PM (Opening time: 9:00 AM, Closing time: 8:00 PM)
• The library remains closed on selected public holidays.

**Frequently Asked Questions:**
• Opening Hours: Open 8:00 AM to 10:00 PM on weekdays (Mon-Fri) and 9:00 AM to 8:00 PM on weekends (Sat-Sun).
• Book Limits: Students can borrow up to 4 books for 14 days.
• Wi-Fi Access: High-speed Wi-Fi is available across all reading halls and workstations.

*(Sources: `library.txt`)*
```

---

### Test Case 4: Hostel Facilities & Amenities
- **Query:** `What are the hostel facilities available?`
- **Tool Routed:** `campus`
- **Canonical Tool Span:** `campus ( query = 'What are the hostel facilities available' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Retrieved Document:** `hostel.txt` (Section: *0. Overview of Hostel Facilities & Residential Amenities*)
- **Expected Final Response:**
```markdown
**Overview of Hostel Facilities & Residential Amenities:**
• Campus student hostels provide fully furnished accommodation and amenities including:
• 2-bed, 3-bed, and 4-bed AC and non-AC residential rooms
• Study desks, ergonomic chairs, personal wardrobes, and high-speed Wi-Fi
• Daily 3-meal dining mess (Breakfast, Lunch, Dinner) with vegetarian and non-vegetarian options
• 24/7 on-campus maintenance for electrical, plumbing, and room facilities
• Safe residential environment with biometric gates, 10:00 PM curfew, and round-the-clock security
• Common study lounges, indoor recreation, and visitor reception rooms

*(Sources: `hostel.txt`)*
```

---

### Test Case 5: Academic Attendance Requirement
- **Query:** `What is the attendance requirement?`
- **Tool Routed:** `campus`
- **Canonical Tool Span:** `campus ( query = 'What is the attendance requirement' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Retrieved Document:** `academic.txt` (Section: *1. Attendance Requirements*)
- **Expected Final Response:**
```markdown
**Attendance Requirements:**
• Students must maintain a minimum of 75% attendance in each registered course to be eligible to write semester examinations.
• Attendance is calculated based on total class sessions attended versus scheduled.
• Students with attendance shortages below 75% require official medical condonation or dean approval according to university regulations.

**Frequently Asked Questions:**
• Minimum Attendance: At least 75% class attendance is required to appear for semester examinations.
• CAT & FAT: CAT 1 & 2 are internal tests; FAT is the final end-semester exam.
• Highest Grade: Grade S is the highest grade with 10 grade points.

*(Sources: `academic.txt`)*
```

---

### Test Case 6: Campus Student Services Overview
- **Query:** `What student services are available on the campus?`
- **Tool Routed:** `campus`
- **Canonical Tool Span:** `campus ( query = 'What student services are available on the campus' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Retrieved Document:** `campus_services.txt` (Section: *0. Overview of Student Services Available on Campus*)
- **Expected Final Response:**
```markdown
**Overview of Student Services Available on Campus:**
• The university provides comprehensive student support services and campus amenities including:
• Medical Centre & 24/7 Emergency Healthcare
• High-Speed Wi-Fi & IT Support Helpdesk
• Free Campus Shuttle Buses & Transit Transportation
• Sports & Gymnasium Facilities (football, cricket, basketball, badminton)
• 50+ Student Clubs & Extracurricular Activities
• Multi-Cuisine Food Courts, Dining Halls & Cafeterias
• Printing, Photocopying & Reprography Kiosks
• 24/7 Central Security & Lost and Found Office

*(Sources: `campus_services.txt`)*
```

---

### Test Case 7: Direct Math Arithmetic
- **Query:** `calculate 125*48`
- **Tool Routed:** `calculator`
- **Canonical Tool Span:** `calculator ( expression = '125*48' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Evaluated Math Result:** `6,000`
- **Expected Final Response:**
```markdown
🧮 Result: `125*48` = **6,000**
```

---

### Test Case 8: Rate & Duration Word Problem
- **Query:** `if i study for 3 hours for 7 days, then how many hours is that?`
- **Tool Routed:** `calculator`
- **Canonical Tool Span:** `calculator ( expression = '3 * 7' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Logic:** Extracts rate $3\text{ hours}$ and duration $7\text{ days}$, forming $3 \times 7 = 21$.
- **Expected Final Response:**
```markdown
🧮 If you study for **3 hours** for **7 days**, that is **21 hours** in total (3 × 7 = 21).
```

---

### Test Case 9: Compound Multi-Domain Campus Query
- **Query:** `What are the library hours and the hostel services available?`
- **Tool Routed:** `campus`
- **Canonical Tool Span:** `campus ( query = 'library hours AND hostel services' )`
- **Execution Step:** Step 7 / 10 (`⚡ EARLY COMMITMENT`)
- **Logic:** Decomposes conjunction `"and"` into sub-queries, retrieving top sections across both `library.txt` and `hostel.txt`.
- **Expected Final Response:**
```markdown
**Library Timings & Opening Hours:**
• Monday to Friday: 8:00 AM to 10:00 PM (Opening time: 8:00 AM, Closing time: 10:00 PM)
• Saturday and Sunday: 9:00 AM to 8:00 PM (Opening time: 9:00 AM, Closing time: 8:00 PM)
• The library remains closed on selected public holidays.

**Overview of Hostel Facilities & Residential Amenities:**
• Campus student hostels provide fully furnished accommodation and amenities including:
• 2-bed, 3-bed, and 4-bed AC and non-AC residential rooms
• Study desks, ergonomic chairs, personal wardrobes, and high-speed Wi-Fi
• Daily 3-meal dining mess (Breakfast, Lunch, Dinner) with vegetarian and non-vegetarian options
• 24/7 on-campus maintenance for electrical, plumbing, and room facilities
• Safe residential environment with biometric gates, 10:00 PM curfew, and round-the-clock security
• Common study lounges, indoor recreation, and visitor reception rooms

**Overview of Student Services Available on Campus:**
• The university provides comprehensive student support services and campus amenities including:
• Medical Centre & 24/7 Emergency Healthcare
• High-Speed Wi-Fi & IT Support Helpdesk
• Free Campus Shuttle Buses & Transit Transportation
• Sports & Gymnasium Facilities (football, cricket, basketball, badminton)
• 50+ Student Clubs & Extracurricular Activities
• Multi-Cuisine Food Courts, Dining Halls & Cafeterias
• Printing, Photocopying & Reprography Kiosks
• 24/7 Central Security & Lost and Found Office

*(Sources: `library.txt`, `hostel.txt`, `campus_services.txt`)*
```

---

## 🚀 How to Run These Test Cases

1. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
2. Paste any of the 9 queries above into the main search box (or click the quick preset buttons).
3. Click **🚀 Run DiffAgent Comparison**.
4. Observe the **Decision Card** (firing at Step 7/10), the **Denoising Heatmap**, the **Live Tool Execution Result**, and the grounded **Final Synthesized Response**.
