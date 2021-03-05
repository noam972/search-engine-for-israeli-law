## **Read me**
מטרת הפרויקט:
 לאתר סעיפים בחקיקה באמצעות סמנטיקה של מילות מפתח
בעת חיפוש רגיל באמצעות מילות מפתח קשה למצוא התאמה כאשר המילים זהות סמנטית אך שונות במבנן. למשל: כאשר נחפש את המילה השמדה ובטקסט תופיע המילה להשמיד נרצה למצוא התאמה אשר לא תופיע עבור חיפוש רגיל.

הקשר למדעי הרוח הדיגיטלים: כחלק מהניסיון להעביר את עולם המשפטים והחקיקה לצורה דיגיטלית, אנו מממשים מנוע חיפוש מתקדם המסוגל להתאים חוק או סעיף בחוק לפי סמנטיקת מילות המפתח או לפי משפט בניגוד למנועי החיפוש הפשוטים המוצאים חוק רק לפי שם החוק או הנושא עליו החוק מדבר. זהו מעבר מעולם מדעי הרוח לעולם הדיגיטלי הנגיש יותר. למשל, בחיפוש הקיים היום כאשר נחפש כשרות לא נמצא חוק איסור גידול חזיר.

תיאור הפרוייקט:
תחילה בשלב העיבוד המוקדם:
תחילה עברנו על כל החוקים בפורמט קבצי XML. איתרנו בכל חוק את הסעיפים שלו כדי שנוכל לעבור עליהם בצורה מסודרת במנוע החיפוש.
טענו רשת WORD2VEC מאומנת המכילה מאגר גדול מאוד של מילים.
טענו את כל הסעיפים על מנת להוציא את הוקטור לכל סעיף לפי TFIDF  
לאחר מכן השתמשנו בשני הכלים לעיל כדי לייצג כל סעיף בעזרת בוקטור.
בעת קבלת הקלט:
בשלב זה, הפעלנו את כל הכלים על מנת לייצר וקטור מתאים לשאילתה שקיבלנו מהמשתמש.
לאחר מכן נשתמש באלגוריתם K-neighbors על מנת למצוא את הסעיפים הקרובים ביותר לשאילתה.

דרך הרצה:
1.להוריד את הקובץ word2vec.model מהגוגל דרייב בקישור הבא: https://drive.google.com/drive/folders/1aDkHoshPPqPQb9EqvBtkpS57jDgmhDRw?usp=sharing
2. להוריד את החוקים בפורמט XML ולשמור בתקיית הפרויקט
2. להתקין את הספריות המתאימות לפרויקט
4. להריץ