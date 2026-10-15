# Netflix Support RAG Assistant — Demo & Interview Q&A (हिंदी में)

> इस फ़ाइल में project के हर हिस्से पर सवाल और छोटे, आसान जवाब हैं।
> Technical terms English में ही रखे हैं, क्योंकि interview में यही बोले जाते हैं।

---

## 0. 30 सेकंड का Pitch (सबसे पहले यही बोलना)

"मैंने Netflix के लिए एक **Customer Support Assistant** बनाया है, नाम है **Ava**। ये **RAG** (Retrieval-Augmented Generation) पर चलता है। मतलब ये पहले Netflix के Help Center documents में से सही जानकारी ढूंढता है, फिर सिर्फ़ उसी जानकारी से जवाब देता है। अपनी तरफ़ से कुछ नहीं बनाता। हर जवाब के साथ ये दिखाता है कि उसने कौन-सा source पढ़ा और क्यों पढ़ा। जिस सवाल की जानकारी documents में नहीं है, उस पर ये politely मना कर देता है और live agent का option देता है। App Streamlit में बनी है, उसमें mock login है, answers live stream होते हैं, follow-up सवाल समझता है, और मैंने इसे benchmark और 38 automated tests से verify किया है।"

---

## 1. Project Overview

**Q1. ये project क्या करता है?**
Netflix customers के billing, account, security और streaming से जुड़े सवालों के जवाब देता है। जवाब सिर्फ़ company के documents से आते हैं, इसलिए गलत या मनगढ़ंत जवाब की संभावना बहुत कम है।

**Q2. Use case क्यों चुना — Customer Support ही क्यों?**
Support में ज़्यादातर सवाल बार-बार आने वाले होते हैं, जैसे "payment fail क्यों हुआ" या "plan कैसे बदलूं"। यहाँ गलत जवाब देना महंगा पड़ता है (refund, legal issue), इसलिए grounded और source दिखाने वाला bot सबसे ज़्यादा काम का है।

**Q3. Business value क्या है?**
Ticket volume कम होता है, जवाब तुरंत मिलता है, brand voice हर बार एक जैसी रहती है, और हर जवाब auditable है क्योंकि source दिखता है।

**Q4. Project के main parts कौन से हैं?**
1. `data/` → knowledge base (4 text documents)
2. `src/ingest.py` → chunking + embedding + vector DB में store
3. `src/local_embeddings.py` → text को vectors में बदलना
4. `src/rag_engine.py` → retrieval + prompt + LLM
5. `src/response_format.py` → reasoning और answer को अलग करना
6. `app.py` → Streamlit UI
7. `eval/benchmark.py` → evaluation
8. `tests/` → automated tests
9. `run.bat` → एक command में setup और launch

**Q5. पूरा flow एक लाइन में बताओ।**
Documents → chunks → embeddings → Chroma DB। User का सवाल → embedding → सबसे मिलते-जुलते 4 chunks → prompt में डालो → LLM (Groq) → पहले "Retrieval Reasoning", फिर "Answer" stream होता है → UI में sources के साथ दिखता है।

---

## 2. RAG — Basics

**Q6. RAG क्या है?**
Retrieval-Augmented Generation। पहले relevant जानकारी **retrieve** करो, फिर उसे LLM को देकर जवाब **generate** करवाओ। मानो LLM को "open book exam" दे दिया हो।

**Q7. Normal LLM (जैसे सीधे ChatGPT) से क्या दिक्कत है?**
उसे Netflix की internal policy पता नहीं होती, उसका knowledge पुराना हो सकता है, और वो confident होकर गलत जवाब (**hallucination**) दे सकता है।

**Q8. RAG hallucination कैसे कम करता है?**
LLM को सिर्फ़ retrieved context दिया जाता है, और prompt में साफ़ लिखा है कि "सिर्फ़ CONTEXT से जवाब दो, नहीं पता तो मना करो।"

**Q9. Fine-tuning की जगह RAG क्यों?**
Fine-tuning महंगा है, धीमा है, और data बदले तो फिर से train करना पड़ता है। RAG में बस document बदलो, re-index करो, काम हो गया। ऊपर से RAG source भी दिखा सकता है, fine-tuned model नहीं दिखा सकता।

**Q10. RAG के दो phases कौन से हैं?**
1. **Indexing (offline):** load → chunk → embed → store।
2. **Querying (online):** question embed → search → prompt → generate।

---

## 3. Embeddings और Vectors

**Q11. Embedding क्या है?**
Text को numbers की list (vector) में बदलना, इस तरह कि जिन texts का मतलब मिलता-जुलता हो, उनके vectors भी पास-पास हों।

**Q12. Vector क्या है?**
Numbers की एक list। हमारे model में हर text एक **384 numbers** का vector बनता है, यानी 384 dimensions।

**Q13. कौन सा embedding model use किया?**
`sentence-transformers/all-MiniLM-L6-v2`। ये छोटा है (~22M parameters), CPU पर चलता है, free है, और semantic search के लिए अच्छा है।

**Q14. ये model क्यों चुना?**
Embedding एक narrow task है, इसके लिए बड़ा model ज़रूरी नहीं। ये local चलता है, इसलिए customer का data बाहर नहीं जाता, API cost भी zero है, और speed भी तेज़ है।

**Q15. "Semantic search" और "keyword search" में क्या फ़र्क है?**
Keyword search सिर्फ़ same शब्द ढूंढता है। Semantic search मतलब समझता है। जैसे "payment declined" और "failed payments" में शब्द अलग हैं, फिर भी match हो जाता है।

**Q16. Mean pooling क्या है?**
Model हर token (शब्द के टुकड़े) का एक vector देता है। हम सारे tokens के vectors का **average** लेकर पूरे sentence का एक vector बनाते हैं। Padding वाले tokens को attention mask से बाहर रखते हैं।

**Q17. L2 normalization क्यों किया?**
हर vector की length 1 कर देते हैं। इससे सिर्फ़ direction मायने रखती है, और cosine similarity सीधे dot product से निकल आती है।

**Q18. Cosine similarity क्या है?**
दो vectors के बीच का angle बताती है। 1 का मतलब बिल्कुल same मतलब, 0 का मतलब कोई संबंध नहीं।

**Q19. Max sequence length 256 क्यों?**
ये model 256 tokens तक के लिए tune किया गया है। हमारे chunks 500 characters के हैं, जो लगभग 100–120 tokens बनते हैं, इसलिए कुछ भी कटता नहीं।

**Q20. Embeddings PyTorch में क्यों नहीं, ONNX में क्यों?**
PyTorch और transformers को import करने में इस machine पर ~30 सेकंड (cold start में 2 मिनट तक) लगते थे, यही login slow होने की असली वजह थी। ONNX Runtime ~0.4 सेकंड में load होता है। Model वही है, vectors बिल्कुल same हैं (cosine = 1.000000), बस चलाने वाला engine हल्का है।

**Q21. ONNX क्या है?**
Open Neural Network Exchange, यानी model को एक standard format में save करने का तरीका, जिसे बिना PyTorch के ONNX Runtime से तेज़ी से चलाया जा सकता है।

**Q22. `tokenizers` library क्या करती है?**
Text को tokens (IDs) में तोड़ती है जो model समझता है। ये Rust में लिखी है, इसलिए बहुत तेज़ है।

**Q23. `local_files_only=True` पहले क्यों try करते हैं?**
Model पहले से cache में है तो internet पर check करने की ज़रूरत नहीं है। पहले Hugging Face का network check अकेले ~45 सेकंड खा रहा था। Cache में न मिले, तभी download होता है।

**Q24. Batch size 32 क्यों?**
बहुत सारे texts एक साथ भेजो तो memory ज़्यादा लगती है। 32-32 के batch में भेजने से speed और memory दोनों balance रहते हैं।

**Q25. `sentence-transformers` library सीधे क्यों नहीं use की?**
वो scipy/sklearn पर depend करती है, और इस machine की Windows **Application Control policy** scipy की एक DLL block कर रही थी। इसलिए हमने वही logic (mean pooling + normalize) खुद लिख दिया।

---

## 4. Chunking

**Q26. Chunking क्या है और क्यों ज़रूरी है?**
बड़े document को छोटे टुकड़ों में तोड़ना। पूरा document एक साथ embed करो तो meaning धुंधला हो जाता है और search सटीक नहीं रहता। छोटे chunks में सटीक हिस्सा मिलता है।

**Q27. Chunk size और overlap कितना है?**
500 characters का chunk, 80 characters का overlap। कुल 38 chunks बने।

**Q28. Overlap क्यों रखा?**
अगर कोई ज़रूरी sentence दो chunks की boundary पर कट जाए, तो overlap की वजह से context दोनों chunks में बचा रहता है।

**Q29. Chunk बहुत छोटा या बहुत बड़ा हो तो क्या होगा?**
बहुत छोटा: context अधूरा, जवाब में जानकारी कम पड़ेगी।
बहुत बड़ा: irrelevant चीज़ें भी साथ आएंगी, search कम सटीक होगा, prompt भी लंबा होगा।

**Q30. `RecursiveCharacterTextSplitter` क्या करता है?**
पहले paragraph (`\n\n`) पर तोड़ने की कोशिश करता है, फिर line (`\n`), फिर sentence (`. `), फिर space पर। यानी जहाँ तक हो सके text को natural जगह से तोड़ता है।

**Q31. `chunk_id` metadata क्यों डाला?**
UI में "chunk #16" दिखाने के लिए, ताकि user source verify कर सके। Files sorted order में load होती हैं, इसलिए IDs हर बार same बनती हैं।

---

## 5. Vector Database — Chroma

**Q32. Vector DB क्या है?**
ऐसा database जो vectors store करता है और "इस vector के सबसे पास वाले vectors कौन से हैं" तेज़ी से बता देता है।

**Q33. Chroma क्यों चुना?**
Free है, local चलता है, कोई server setup नहीं चाहिए, disk पर persist होता है, और LangChain के साथ आसानी से जुड़ता है। छोटे project के लिए बिल्कुल सही है।

**Q34. Data कहाँ store होता है?**
`chroma_db/` folder में, SQLite file और HNSW index के रूप में।

**Q35. HNSW क्या है?**
Hierarchical Navigable Small World, एक fast approximate nearest-neighbour search algorithm। ये हर vector से compare नहीं करता, एक graph के ज़रिए जल्दी से सबसे पास वाले vectors तक पहुँच जाता है।

**Q36. Collection में "cosine" space क्यों set किया?**
पहले default **L2 distance** था, जिससे relevance scores negative (जैसे -0.08) आ रहे थे और warning भी आती थी। हमारे vectors normalized हैं, इसलिए cosine सही metric है। अब scores 0 से 1 के बीच आते हैं।

**Q37. अगर cosine हटा दें तो क्या होगा?**
Ranking फिर भी लगभग सही रहेगी, लेकिन UI में दिखने वाले relevance scores गलत और confusing होंगे, और low-confidence warning भी गलत काम करेगी।

**Q38. Alternatives क्या थे?**
FAISS (तेज़ है, पर persistence और metadata खुद संभालने पड़ते हैं), Pinecone/Weaviate (cloud, scalable, पर setup और cost लगते हैं), pgvector (Postgres में)।

---

## 6. Retrieval

**Q39. Retrieval कैसे होता है?**
सवाल का embedding बनता है, फिर Chroma में सबसे ज़्यादा similar **top_k = 4** chunks निकलते हैं, relevance score के साथ।

**Q40. Top-k 4 ही क्यों?**
4 chunks से पर्याप्त context मिल जाता है और prompt छोटा रहता है। ज़्यादा लो तो noise बढ़ता है, कम लो तो ज़रूरी जानकारी छूट सकती है। User sidebar में इसे 1 से 8 तक बदल सकता है।

**Q41. Relevance score क्या है?**
सवाल और chunk कितने मिलते-जुलते हैं, इसका 0 से 1 के बीच का score। UI में % और progress bar के रूप में दिखता है।

**Q42. Low-confidence warning कब आती है?**
जब सबसे अच्छे match का score भी **0.30 से कम** हो। हमने इसे data से calibrate किया: सारे Netflix वाले सवाल ≥ 0.38 आए और बिल्कुल off-topic सवाल (pasta recipe, France की capital) ≤ 0.14।

**Q43. "Netflix का stock price क्या है?" पर score 0.62 क्यों आता है? Warning क्यों नहीं आती?**
Topic Netflix का है, इसलिए words match होते हैं। ऐसे cases में दूसरी सुरक्षा परत काम करती है: LLM prompt के हिसाब से मना कर देता है क्योंकि context में stock price नहीं है।

**Q44. Context LLM को किस format में जाता है?**
हर chunk के आगे `[Source: filename]` लगाकर। इससे LLM अपनी reasoning में filename का reference दे पाता है।

---

## 7. Model Selection — LLM vs SLM

**Q45. LLM और SLM में क्या फ़र्क है?**
LLM = Large Language Model (अरबों parameters, ज़्यादा smart, महंगा)। SLM = Small Language Model (कम parameters, सस्ता, तेज़, लेकिन कमज़ोर reasoning)।

**Q46. आपने कहाँ SLM और कहाँ LLM use किया?**
- **Embedding → SLM** (MiniLM): ये एक narrow task है, छोटा model काफ़ी है।
- **Generation → LLM** (gpt-oss-120b): persona follow करना, reasoning लिखना और सही जगह मना करना, इन सबके लिए strong instruction-following चाहिए।

**Q47. Generation के लिए SLM क्यों नहीं?**
छोटे models (जैसे Phi-3-mini, Llama-3.2-3B) अक्सर "context में नहीं है तो मना करो" वाला rule तोड़ देते हैं या persona से बाहर चले जाते हैं। Support bot में hallucination का risk ज़्यादा महंगा है।

**Q48. Rule of thumb क्या है?**
"सबसे छोटा model जो task को reliably कर दे।"

**Q49. कौन सा LLM use किया?**
`openai/gpt-oss-120b`, जो **Groq** पर host है। ये OpenAI का open-weight model है।

**Q50. Groq क्यों?**
Groq का LPU hardware बहुत तेज़ inference देता है, जो chat app में ज़रूरी है। Free tier भी अच्छा है। API OpenAI-compatible है।

**Q51. पहले कौन सा model plan था, बदला क्यों?**
पहले `llama-3.3-70b-versatile` था, लेकिन Groq पर वो अब उपलब्ध नहीं था (404 "model not found")। इसलिए Groq की available models की list निकालकर gpt-oss-120b चुना।

**Q52. Provider बदलना हो तो?**
`.env` में `LLM_PROVIDER=openai` और `OPENAI_API_KEY` डालो। Code में दोनों के लिए `_build_llm()` function है, बाकी कुछ बदलना नहीं पड़ता।

**Q53. `reasoning_effort="low"` क्या है?**
gpt-oss model जवाब से पहले अंदर ही अंदर (hidden) सोचता है। हमारा prompt पहले से reasoning बाहर लिखवाता है, इसलिए अंदर की reasoning कम कर दी। इससे पहला word आने का समय ~1.6s से ~0.6s हो गया, और benchmark फिर भी 21/21 रहा।

---

## 8. Prompt Engineering

**Q54. System prompt में क्या-क्या है?**
1. **Persona:** "Ava", Netflix की warm, clear support assistant।
2. **Brand voice:** customer को blame नहीं करना, jargon नहीं, जल्दी solution।
3. **Strict grounding:** सिर्फ़ CONTEXT से जवाब।
4. **Refusal line:** "I don't have that information... live agent..."
5. **Authorization limits:** legal commitment नहीं, refund guarantee नहीं, unreleased content पर बात नहीं।
6. **Chain-of-Thought format:** "Retrieval Reasoning:" फिर "Answer:"।

**Q55. Persona क्यों ज़रूरी है?**
Brand की आवाज़ हर जवाब में एक जैसी रहे, professional और friendly। Assignment में भी "strict brand-voice persona" माँगा गया था।

**Q56. Chain-of-Thought (CoT) क्या है?**
Model से पहले उसकी सोच लिखवाना, फिर final answer। हमारे case में model पहले बताता है कि कौन सा chunk क्यों use किया, फिर customer को जवाब देता है।

**Q57. CoT से फ़ायदा क्या?**
Retrieval logic audit हो सकता है (गलत source लिया तो दिख जाएगा), और step-by-step सोचने से जवाब ज़्यादा सटीक होता है।

**Q58. Exact refusal sentence क्यों fix किया?**
ताकि behaviour predictable रहे और benchmark में उसे पकड़ा जा सके।

**Q59. Prompt injection से कैसे बचाव है?**
System prompt में strict rules हैं और persona से बाहर जाने पर मनाही है। फिर भी ये 100% guarantee नहीं है, ये एक limitation है।

**Q60. ChatPromptTemplate क्या है?**
LangChain का template, जिसमें system message, history और user question की जगहें तय होती हैं, और runtime पर values भर दी जाती हैं।

---

## 9. Generation Parameters

**Q61. Temperature क्या है?**
Randomness का control। 0 पर हर बार लगभग same, सबसे संभावित जवाब आता है। 0.8 पर जवाब ज़्यादा creative और varied होते हैं।

**Q62. Top-P क्या है?**
Nucleus sampling। Model सिर्फ़ उन words में से चुनता है जिनकी cumulative probability P तक पहुँचती है। 0.9 का मतलब नीचे के 10% unlikely words हटा दो।

**Q63. तीन presets कौन से हैं?**
- `factual_strict`: temp 0.0, top_p 0.9
- `balanced`: temp 0.3, top_p 0.9
- `creative`: temp 0.8, top_p 0.95

**Q64. किस setting से hallucination सबसे कम हुआ?**
Automated test में तीनों ने 7/7 pass किया, क्योंकि ये model काफ़ी strong है। फिर भी **temperature 0.0** recommend किया, क्योंकि उसमें phrasing सबसे consistent है और जवाब source के शब्दों के सबसे करीब रहता है। Creative वाला policy details को paraphrase करता है, जिसमें risk ज़्यादा है।

**Q65. Sliders बदलने पर क्या होता है?**
पहले पूरा engine दोबारा बनता था (धीमा था)। अब embedding और DB shared रहते हैं, और LLM client `lru_cache` से cache होता है, इसलिए setting बदलना instant है।

---

## 10. Conversation Memory और Follow-ups

**Q66. Follow-up सवाल कैसे समझता है?**
दो काम होते हैं:
1. **Question condensing:** "and if I downgrade?" को LLM एक standalone सवाल में बदलता है, जैसे "If I downgrade my plan, when does it take effect?"। Search इसी से होता है।
2. **History in prompt:** पिछली 3 exchanges (6 messages) भी prompt में जाती हैं।

**Q67. Condensing क्यों ज़रूरी है? सीधे "and if I downgrade?" से search करो तो?**
उस छोटे सवाल में "plan" शब्द ही नहीं है, तो गलत chunks आ सकते हैं। Standalone सवाल से सही chunk मिलता है।

**Q68. History में assistant की reasoning भी जाती है?**
नहीं, सिर्फ़ final "Answer" जाता है। इससे tokens बचते हैं और model अपनी पुरानी reasoning दोहराता नहीं।

**Q69. History सिर्फ़ 3 turns ही क्यों?**
ज़्यादा history से prompt लंबा होता है, cost और latency बढ़ती है, और पुरानी बातें confuse कर सकती हैं। Support chat में 3 turns काफ़ी हैं।

**Q70. क्या history से model facts उठा सकता है?**
नहीं, prompt में साफ़ लिखा है कि history सिर्फ़ follow-up समझने के लिए है, facts CONTEXT से ही आने चाहिए।

**Q71. Memory बंद कैसे करें?**
Sidebar में "Remember conversation" का toggle है। बंद करो तो हर सवाल independent हो जाता है।

**Q72. Condensing fail हो जाए तो?**
`try/except` है। Error आए तो original सवाल ही use होता है, app नहीं टूटती। पहले सवाल पर (जब history नहीं होती) condensing call होती ही नहीं, ताकि समय बचे।

---

## 11. Streaming और "Thinking" UI

**Q73. Streaming क्या है?**
पूरा जवाब बनने का इंतज़ार करने की बजाय, LLM जैसे-जैसे शब्द बनाता है वैसे-वैसे screen पर दिखाना। User को लगता है जवाब तुरंत आ रहा है।

**Q74. Code में streaming कैसे होती है?**
`chain.stream(...)` एक generator देता है जो text के टुकड़े yield करता है। App हर टुकड़ा buffer में जोड़कर placeholder (`st.empty()`) को update करती है।

**Q75. Reasoning और Answer अलग-अलग कैसे दिखते हैं?**
`response_format.py` में regex है जो "Answer:" header ढूंढता है। उससे पहले का text "thinking" panel में जाता है और उसके बाद का text main answer में। Streaming के दौरान भी ये हर update पर चलता है।

**Q76. Regex "Answering this…" जैसे शब्द को header समझ ले तो?**
नहीं समझता। Regex में word boundary (`\b`) और colon (या line का end) ज़रूरी है। इसके लिए test भी लिखा है।

**Q77. Model "Answer:" header लिखना भूल जाए तो?**
`final_sections()` पूरे text को answer मान लेता है। जवाब फिर भी दिखेगा, बस reasoning panel खाली रहेगा।

**Q78. TTFT क्या है?**
Time To First Token, यानी सवाल भेजने से पहला शब्द दिखने तक का समय। UI में "first token Xs · answered in Ys" दिखता है।

**Q79. Throttle (0.04s) क्यों है?**
हर एक token पर screen update करो तो browser पर load बढ़ता है। Answer phase में हर 40ms में एक बार update करते हैं, आखिर में final render ज़रूर होता है।

**Q80. `st.status` क्या है?**
एक collapsible box जिसका label बदलता है: "Thinking" → "Searching knowledge base" → "Reasoning over 4 sources" → complete। User को pipeline का हर step दिखता है।

---

## 12. Streamlit App और Features

**Q81. Streamlit क्यों?**
Python में ही UI बन जाती है, HTML/JS नहीं लिखना पड़ता, और chat elements (`st.chat_message`, `st.chat_input`) पहले से मौजूद हैं। Demo और prototype के लिए perfect है।

**Q82. Streamlit का execution model क्या है?**
हर interaction (button click, input) पर **पूरी script ऊपर से नीचे दोबारा चलती है**। Data बचाने के लिए `st.session_state` use होता है।

**Q83. `st.session_state` में क्या-क्या रखा है?**
`authenticated`, `messages` (पूरी chat), `feedback_<index>`, और `pending_query`।

**Q84. Retrieved Chunks panel क्यों?**
Assignment की requirement है कि user source verify कर सके। इसमें हर chunk का filename, chunk number, relevance score, progress bar और पूरा text दिखता है।

**Q85. Sources line क्या है?**
Answer के नीचे एक लाइन, जैसे "📎 Sources: 02_account_and_security.txt (74%)"। हर file का best score दिखता है, ताकि एक नज़र में source पता चल जाए।

**Q86. Sample question chips कैसे काम करते हैं?**
Button दबाने पर सवाल `pending_query` में जाता है, और उसी run में `chat_input` खाली होने पर वही सवाल process हो जाता है।

**Q87. Regenerate button क्या करता है?**
आखिरी user सवाल और उसका जवाब हटाता है, उस message का feedback भी साफ़ करता है, फिर वही सवाल दोबारा भेजता है। ये सिर्फ़ आखिरी जवाब पर दिखता है।

**Q88. Feedback (👍/👎) कहाँ जाता है?**
`logs/feedback.jsonl` में: timestamp, rating, सवाल, जवाब, sources, parameters और model। Same rating दोबारा दबाने पर दोबारा log नहीं होता।

**Q89. JSONL क्यों?**
हर line एक JSON record है। File में बस append करते जाओ, पूरी file दोबारा लिखनी नहीं पड़ती। Analysis के लिए pandas से आसानी से पढ़ी जा सकती है।

**Q90. Feedback log fail हो जाए (disk full वगैरह) तो?**
`except OSError: pass` है। Logging की वजह से chat कभी नहीं टूटती।

**Q91. Clear chat में feedback भी क्यों साफ़ करते हैं?**
ये एक bug था। Chat साफ़ होने के बाद नया जवाब उसी index पर आता था और पुराना 👍 highlight दिख जाता था। अब `feedback_*` keys भी delete होती हैं।

**Q92. Session stats में क्या दिखता है?**
कितने सवाल पूछे, average response time, average first-token time, और 👍/👎 की गिनती।

**Q93. Knowledge Base browser क्या है?**
Sidebar में सारी source files पढ़ सकते हो, और "Re-index" button से index दोबारा बना सकते हो। File पढ़ने को `st.cache_data` से cache किया है, और file का modification time (mtime) cache key में है, ताकि file edit होने पर नया text दिखे।

**Q94. Transcript download में क्या आता है?**
Markdown file, जिसमें हर customer सवाल, Ava का जवाब (reasoning के बिना) और sources होते हैं।

**Q95. Theme कैसे set किया?**
`.streamlit/config.toml` में Netflix जैसा dark theme और red primary color (#E50914)। Usage stats भेजना भी बंद किया।

---

## 13. Login और Security

**Q96. Mock login कैसे काम करता है?**
Username और password `.env` से आते हैं। Login सही होने पर `session_state.authenticated = True` होता है। जब तक login नहीं, `st.stop()` बाकी app को चलने ही नहीं देता।

**Q97. `hmac.compare_digest` क्यों use किया, `==` क्यों नहीं?**
`==` पहला गलत character मिलते ही रुक जाता है, जिससे timing देखकर password guess करने का (timing attack) risk रहता है। `compare_digest` हमेशा बराबर समय लेता है।

**Q98. `.encode()` क्यों किया?**
`compare_digest` में non-ASCII string (जैसे "ädmin") डालो तो वो error देता है। Bytes में बदलने से crash नहीं होता। इसका test भी है।

**Q99. Logout पर क्या होता है?**
`st.session_state.clear()` से पूरी chat, feedback, सब कुछ साफ़ होता है। पहले ये bug था कि logout के बाद अगले login पर पुरानी chat दिख जाती थी।

**Q100. ये real authentication क्यों नहीं है?**
Password hashing, user database, session expiry, rate limiting, कुछ नहीं है। ये सिर्फ़ demo के लिए "mock" है। Production में Auth0, OAuth या Streamlit का native auth use करना चाहिए।

**Q101. API key कहाँ रखी है?**
`.env` file में, जो `.gitignore` में है, यानी GitHub पर नहीं जाती। `.env.example` में सिर्फ़ placeholder है।

**Q102. Bot sensitive data माँगता है क्या?**
नहीं। Knowledge base की policy कहती है कि full card number, password या 2FA code कभी नहीं माँगना।

---

## 14. Performance Optimization (बहुत ज़रूरी सवाल)

**Q103. पहले login में 45 सेकंड क्यों लगते थे?**
Login होते ही engine बनता था। उसमें PyTorch और transformers import होते थे (~30s warm, cold में 2 मिनट तक), ऊपर से Hugging Face का network check भी होता था।

**Q104. अब क्या किया?**
1. PyTorch की जगह **ONNX Runtime** (import ~0.4s)।
2. Model पहले **local cache** से load होता है।
3. **Background warm-up thread:** user जब तक login screen पर username-password टाइप कर रहा होता है, तब तक engine पीछे load हो जाता है।
4. Embedding और DB **shared** हैं (हर session में एक ही बार बनते हैं)।
5. Login के बाद page engine का इंतज़ार ही नहीं करता, engine सिर्फ़ पहले सवाल पर चाहिए।
6. Groq पर `reasoning_effort="low"`।

**Q105. Result क्या रहा?**
Login → chat page: **0.06s**। Engine background में ~3s में तैयार। पहला answer token ~0.6–1.9s।

**Q106. `@st.cache_resource` क्या करता है?**
Function का result पूरी server process के लिए एक बार बनाकर रखता है, सारे users और reruns में shared। हमने इससे warm-up thread सिर्फ़ एक बार start किया है।

**Q107. Lock (`threading.Lock`) क्यों लगाया?**
Background thread और user की request एक ही समय पर model load करने लगें तो model दो बार load होगा। Lock से एक समय पर एक ही load होता है, और दूसरा तैयार object ले लेता है।

**Q108. `lru_cache` कहाँ use किया?**
`_build_llm(temperature, top_p)` पर। Same setting दोबारा आए तो नया client नहीं बनता।

**Q109. Sidebar में 🟢/🟡/🔴 क्या है?**
Engine का status: 🟢 ready, 🟡 warming up, 🔴 warm-up fail (पहले सवाल पर दोबारा try होगा)।

**Q110. Server बिल्कुल अभी start हुआ हो और user तुरंत सवाल पूछ दे तो?**
तब पहला सवाल warm-up खत्म होने तक (~3–6s) रुकेगा और "Loading AI models (first run only)" दिखेगा। उसके बाद सब तेज़ चलेगा।

---

## 15. Knowledge Base Auto-Sync

**Q111. नई file `data/` में डालें तो क्या करना पड़ेगा?**
कुछ नहीं। App start होने पर सारी files का **SHA-256 fingerprint** compare होता है, और बदलाव मिले तो index अपने आप rebuild हो जाता है। चाहो तो sidebar का "Re-index" button भी दबा सकते हो।

**Q112. Fingerprint में क्या-क्या आता है?**
Chunk size, overlap, हर file का नाम और content। इनमें से कुछ भी बदले तो fingerprint बदल जाता है।

**Q113. Fresh GitHub clone पर `chroma_db` नहीं होता (gitignore में है), तब क्या?**
App खुद detect करती है कि DB खाली है और index बना देती है (~10s)। पहले इस case में bot हर सवाल पर "don't have that information" बोलता था।

**Q114. Rebuild के लिए folder delete क्यों नहीं करते?**
Windows पर Chroma की SQLite file खुली रहती है, तो folder delete या rename करने पर "Access is denied" आता है। इसलिए हम collection के अंदर के records delete करके नए डालते हैं (in-place)। ये app चलते हुए भी काम करता है।

---

## 16. Evaluation / Benchmark

**Q115. Benchmark कैसे किया?**
`eval/benchmark.py`: 7 सवाल (4 जिनका जवाब KB में है, 3 जिन पर मना करना चाहिए: stock price, नई series, किसी और का password) × 3 presets = 21 live runs। Result `eval/results.csv` में save होता है।

**Q116. Pass/fail कैसे तय होता है?**
Answerable सवाल पर: जवाब में refusal phrase नहीं होना चाहिए। Unanswerable सवाल पर: refusal phrase होना चाहिए ("don't have that information", "can't provide", "can't share" वगैरह)।

**Q117. "live agent" को marker क्यों हटाया?**
अच्छे जवाब भी "live agent से connect कर सकते हैं" बोलते हैं, जैसे refund वाले जवाब में। इससे सही जवाब भी fail गिने जा रहे थे। ये bug real transcripts देखकर पकड़ा और fix किया।

**Q118. Apostrophe normalize क्यों किया?**
LLM कभी curly ’ लिखता है, कभी straight '। Normalize न करो तो "don’t" और "don't" match नहीं होते और result गलत आता है।

**Q119. Result क्या आया?**
तीनों presets में **7/7**, कुल 21/21।

**Q120. इस evaluation की limitation क्या है?**
Sample छोटा है (7 सवाल), और ये सिर्फ़ "मना किया या जवाब दिया" check करता है, ये नहीं कि जवाब कितना सही (faithful) है। Next step: बड़ा question set और **LLM-as-judge** या RAGAS जैसे metrics (faithfulness, answer relevance, context precision)।

---

## 17. Testing

**Q121. कितने tests हैं, क्या-क्या cover होता है?**
38 pytest tests: response parsing, benchmark scoring, chunking और fingerprint, embeddings (384-dim, normalized), retrieval accuracy (सही document top पर), history trimming, और UI tests।

**Q122. UI tests कैसे किए?**
Streamlit के **AppTest** framework से, जो बिना browser के app चलाकर button click और input simulate करता है। इसमें login, गलत password, non-ASCII input, streaming answer, sample chip, feedback, regenerate, error handling, logout और transcript सब check होते हैं।

**Q123. Tests में LLM call होती है?**
नहीं। LLM की जगह एक fake generator लगाया है (monkeypatch), जो fixed text देता है। इससे tests offline, तेज़ और हर बार same result देते हैं। Live LLM की जाँच benchmark से होती है।

**Q124. Tests कैसे चलाएं?**
`python -m pytest tests`

---

## 18. Problems जो रास्ते में आए (Debugging stories)

**Q125. सबसे बड़ी problem क्या आई?**
1. Python 3.14 बहुत नया है, कई पुराने package versions के wheels नहीं थे (pandas/numpy build fail हुए)। Versions unpin करके हल किया।
2. Windows Application Control ने scipy की DLL block की, तो sentence-transformers छोड़कर अपना embedder लिखा।
3. Llama model Groq से हट चुका था (404), तो gpt-oss-120b पर shift किया।
4. Negative relevance scores आए (L2 vs cosine), cosine set करके fix किया।
5. Login 45s ले रहा था, ONNX और background warm-up से fix किया।

**Q126. API key गलत हो तो क्या दिखता है?**
App crash नहीं होती। Friendly message दिखता है: "AI service rejected the API key…", और नीचे "Error details" में असली error मिलती है।

**Q127. कौन-कौन सी errors handle होती हैं?**
Invalid API key (401), rate limit (429), model not found, internet/connection error, empty response, और key missing।

**Q128. Error के बाद chat history का क्या होता है?**
User का सवाल और एक error message दोनों save होते हैं, history अधूरी नहीं रहती। Regenerate से दोबारा try कर सकते हो। Error वाले messages अगली बार LLM की history में नहीं भेजे जाते।

**Q129. Groq पर कभी-कभी 5–7 सेकंड क्यों लगते हैं?**
Free tier की **rate limit** (tokens per minute)। लगातार बहुत सवाल पूछो तो Groq धीमा कर देता है। ये code की problem नहीं है। Client में `max_retries=2` रखा है।

---

## 19. Limitations

**Q130. इस project की limitations क्या हैं?**
1. Knowledge base छोटा और simulated है (4 docs), असली Netflix data नहीं है।
2. Login सिर्फ़ mock है।
3. Evaluation छोटा है, faithfulness score नहीं है।
4. Groq free tier rate limit।
5. सिर्फ़ English documents।
6. Prompt injection से पूरी सुरक्षा नहीं।
7. एक machine पर local चलता है, scale के लिए deploy करना होगा।
8. Text files ही support होती हैं (PDF/HTML नहीं)।

**Q131. Bot कभी गलत जवाब दे सकता है?**
हाँ। अगर retrieval गलत chunk लाए, या document में ही गलत जानकारी हो। इसीलिए sources और reasoning दिखाते हैं ताकि इंसान verify कर सके।

---

## 20. Alternatives और Future Improvements

**Q132. Retrieval और बेहतर कैसे कर सकते हैं?**
- **Hybrid search:** BM25 keyword search + vector search (error codes जैसे "NW-2-5" exact match में बेहतर)।
- **Re-ranker** (cross-encoder) से top results को दोबारा rank करना।
- Metadata filters (जैसे सिर्फ़ billing docs में search)।

**Q133. बड़े scale पर क्या बदलेंगे?**
Chroma की जगह managed vector DB (Pinecone, pgvector), API के लिए FastAPI backend, caching, real auth और monitoring।

**Q134. Feedback का आगे क्या use है?**
👎 वाले और low-confidence सवाल निकालकर पता चलेगा कि knowledge base में किस topic की कमी है, फिर वो documents जोड़ दो।

**Q135. LangChain की जगह क्या use कर सकते थे?**
LlamaIndex, या बिना framework के सीधे Groq SDK + chromadb। LangChain से prompt templates, streaming और vector store integration आसानी से मिल गए।

**Q136. Multilingual (Hindi) support कैसे देंगे?**
Multilingual embedding model (जैसे `paraphrase-multilingual-MiniLM`) और prompt में "customer की भाषा में जवाब दो" जोड़कर।

---

## 21. Code-Level Cross Questions

**Q137. `sys.path.insert(0, "src")` app.py में क्यों है?**
ताकि `config`, `rag_engine` जैसे modules सीधे import हो सकें, क्योंकि वो `src/` folder में हैं।

**Q138. `load_dotenv(PROJECT_ROOT/.env)` में पूरा path क्यों दिया?**
App किसी भी folder से launch हो, `.env` तब भी मिल जाए। सिर्फ़ `load_dotenv()` current folder में ढूंढता है।

**Q139. `rag_engine` को app.py में ऊपर import क्यों नहीं किया?**
Top पर import करते तो login page भी heavy libraries load होने तक रुकता। अब वो सिर्फ़ background thread में और पहले सवाल पर import होता है।

**Q140. `st.stop()` का role क्या है?**
Login नहीं हुआ तो script वहीं रुक जाती है, नीचे का कोई code (sidebar, chat, engine) चलता ही नहीं। यही security gate है।

**Q141. `st.rerun()` सवाल के बाद क्यों?**
Streamlit ऊपर से नीचे चलता है। Sidebar (stats, download button) पहले render हो चुका होता है, जब नया message add होता है। Rerun से पूरा page नई state के साथ दोबारा बनता है।

**Q142. `MessagesPlaceholder("history", optional=True)` क्या है?**
Prompt में वो जगह जहाँ पिछली chat डाली जाती है। `optional` का मतलब है कि history न हो तब भी prompt काम करे।

**Q143. `StrOutputParser` क्या करता है?**
LLM के message object में से सिर्फ़ text निकालता है, ताकि हमें plain string मिले (streaming में भी)।

**Q144. `similarity_search_with_relevance_scores` vs `similarity_search`?**
पहला वाला documents के साथ 0–1 का score भी देता है, जो हमें UI और low-confidence check के लिए चाहिए।

**Q145. Chunks के ids `chunk-0`, `chunk-1` क्यों दिए?**
Stable IDs रहें और re-index पर duplicates न बनें।

**Q146. `daemon=True` thread क्यों?**
Server बंद हो तो ये background thread program को रोके नहीं, अपने आप बंद हो जाए।

**Q147. `run.bat` क्या करता है?**
venv न हो तो बनाता है और dependencies install करता है, `.env` न हो तो `.env.example` से बना देता है, फिर `streamlit run app.py` चलाता है। एक command में पूरा setup।

**Q148. requirements में version bounds (`>=1.40,<2`) क्यों?**
ताकि भविष्य में कोई बड़ा (breaking) version आए तो install न हो, और project टूटे नहीं।

**Q149. Torch requirements से हटा दिया, तो कुछ टूटा?**
नहीं। Fresh environment में बिना torch के install करके सारे 38 tests pass किए। उल्टा, install ~2 GB छोटा हो गया।

---

## 22. Demo के दौरान Practical सवाल

**Q150. Demo में क्या-क्या दिखाऊं? (Order)**
1. Login screen → पहले गलत password (error दिखेगा), फिर सही password।
2. Sidebar में 🟢 engine ready दिखाओ।
3. Sample chip "Why was my payment declined?" दबाओ → thinking panel → answer stream।
4. Sources line और Retrieved Chunks खोलकर score दिखाओ।
5. Follow-up पूछो: "How do I change my plan?" फिर "and if I downgrade, when does it apply?" → "Searched for…" दिखाओ।
6. Off-topic पूछो: "What is the capital of France?" → refusal + ⚠️ low confidence।
7. Policy test: "Can you give me someone else's password?" → मना करता है।
8. Preset को creative करो और फ़र्क दिखाओ।
9. 👍 दबाओ, Session stats दिखाओ, transcript download करो।
10. KB browser और Re-index दिखाओ।

**Q151. "Live दिखाओ कि ये hallucinate नहीं करता।"**
ऐसा सवाल पूछो जिसका जवाब docs में नहीं है, जैसे "What's the price of the Premium plan in India?"। Documents में कीमतें नहीं हैं ("prices vary by country"), तो bot मना करेगा या local currency वाली बात बताएगा, कोई number खुद से नहीं बनाएगा।

**Q152. "अगर मैं document बदल दूं तो?"**
`data/` में text बदलो → Re-index दबाओ (या app restart करो) → नया जवाब नई जानकारी से आएगा।

**Q153. "Internet बंद हो तो?"**
Embedding और search local हैं, वो चलते रहेंगे। LLM (Groq) cloud पर है, तो जवाब नहीं आएगा और friendly connection error दिखेगा।

**Q154. "ये कितना cost करता है?"**
Embedding free (local)। Groq का free tier demo के लिए काफ़ी है। Production में per-token cost लगेगी।

**Q155. "Data privacy का क्या?"**
Embedding local होती है। Groq पर सिर्फ़ सवाल और retrieved chunks जाते हैं। Real deployment में data processing agreement और PII masking जोड़ना चाहिए।

---

## 23. एक लाइन में याद रखने वाली बातें (Quick Revision)

| Topic | याद रखो |
|---|---|
| Embedding model | all-MiniLM-L6-v2, 384-dim, ONNX, local |
| LLM | openai/gpt-oss-120b on Groq, reasoning_effort low |
| Vector DB | Chroma, cosine, 38 chunks |
| Chunking | 500 chars, 80 overlap, Recursive splitter |
| Retrieval | top_k = 4, low-confidence < 0.30 |
| Memory | last 3 exchanges + question condensing |
| Presets | 0.0/0.9, 0.3/0.9, 0.8/0.95 |
| Benchmark | 7 सवाल × 3 presets = 21/21 pass |
| Tests | 38 pytest (AppTest से UI भी) |
| Login speed | 45s → 0.06s |
| First token | ~0.6–1.9s |
| Run | `.\run.bat`, login admin / netflix123 |
