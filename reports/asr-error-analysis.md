# ASR error analysis

> Evidence only. This report does not rank models or pick a winner. Weigh quality, latency, hardware, concurrency and licensing for the actual use case. Numbers come only from runs listed below.

- Run: `/home/digital-nepal/Documents/dn-ai/nepali-voice-agent-lab/results/asr/20260923T102131574Z-asr`
- Dataset: `data/manifests/asr_synthetic_piper-ne-google-x-low.jsonl` (sha c6f0b7694bfa)
- Hardware: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz (4 threads), 16.1 GB RAM, GPU: none (CPU-only)

## indicconformer-600m-rnnt [original]

Error tag counts (samples containing each error type): substitution=21, devanagari=15, code_switching=9, english_word=9, missed:mixed_language=9, insertion=5, number=4, deletion=3, missed:names=3, missed:currency=2, missed:addresses=2, missed:phone_numbers=1, missed:otps=1, missed:account_numbers=1, missed:numbers=1, missed:dates=1

### substitution

- **p-basic-01** (basic_nepali)
  - Prediction: नमस्ते मलाई तपाईँको सहयोग चाहियो
  - Word errors: तपाईंको→तपाईँको

- **p-devanagari-01** (devanagari)
  - Prediction: कृपया मेरो गुना सुधर्ता गरिदिनुहोस्
  - Word errors: गुनासो→गुना; दर्ता→सुधर्ता

- **p-conjunct-02** (conjunct_characters)
  - Prediction: श्रीमा स्वास्थ्य परीक्षण द्वितीय पटक गरियो
  - Word errors: श्रीमान्को→श्रीमा

### devanagari

- **p-basic-01** (basic_nepali)
  - Prediction: नमस्ते मलाई तपाईँको सहयोग चाहियो
  - Word errors: तपाईंको→तपाईँको

- **p-devanagari-01** (devanagari)
  - Prediction: कृपया मेरो गुना सुधर्ता गरिदिनुहोस्
  - Word errors: गुनासो→गुना; दर्ता→सुधर्ता

- **p-conjunct-02** (conjunct_characters)
  - Prediction: श्रीमा स्वास्थ्य परीक्षण द्वितीय पटक गरियो
  - Word errors: श्रीमान्को→श्रीमा

### code_switching

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-currency-01** (currency)
  - Prediction: मेरो बास पच्चिस हजार छ
  - Word errors: balance→बास; रु→पच्चिस; 25000→हजार

### english_word

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-currency-01** (currency)
  - Prediction: मेरो बास पच्चिस हजार छ
  - Word errors: balance→बास; रु→पच्चिस; 25000→हजार

### missed:mixed_language

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-currency-01** (currency)
  - Prediction: मेरो बास पच्चिस हजार छ
  - Word errors: balance→बास; रु→पच्चिस; 25000→हजार

### insertion

- **p-numbers-01** (phone_numbers)
  - Prediction: मेरो फोन नम्बर नभो अरब चौरासी करोड बाह्र लाख चौतिस हजार पाँच सय सतसट्ठी हो
  - Word errors: ∅→नभो; ∅→अरब; ∅→चौरासी; ∅→करोड; ∅→बाह्र; ∅→लाख; ∅→चौतिस; ∅→हजार

- **p-numbers-03** (account_numbers)
  - Prediction: मेरो खाता नम्बर शून्य एक दुई तिन चार पाँच छ सात आठ नौ हो
  - Word errors: ∅→शून्य; ∅→एक; ∅→दुई; ∅→तिन; ∅→चार; ∅→पाँच; ∅→छ; ∅→सात

- **p-dates-01** (dates)
  - Prediction: मेरो पृथ्वी मन्त्र दुरी हजार त्रासी साल असोस बार गति हो
  - Word errors: ∅→पृथ्वी; ∅→मन्त्र; ∅→दुरी; appointment→हजार; 2083→त्रासी; असोज→असोस; 12→बार; गते→गति

### number

- **p-numbers-01** (phone_numbers)
  - Prediction: मेरो फोन नम्बर नभो अरब चौरासी करोड बाह्र लाख चौतिस हजार पाँच सय सतसट्ठी हो
  - Word errors: ∅→नभो; ∅→अरब; ∅→चौरासी; ∅→करोड; ∅→बाह्र; ∅→लाख; ∅→चौतिस; ∅→हजार

- **p-numbers-03** (account_numbers)
  - Prediction: मेरो खाता नम्बर शून्य एक दुई तिन चार पाँच छ सात आठ नौ हो
  - Word errors: ∅→शून्य; ∅→एक; ∅→दुई; ∅→तिन; ∅→चार; ∅→पाँच; ∅→छ; ∅→सात

- **p-currency-01** (currency)
  - Prediction: मेरो बास पच्चिस हजार छ
  - Word errors: balance→बास; रु→पच्चिस; 25000→हजार

### deletion

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-english-01** (english)
  - Prediction: ए पाउँदा तत्साक नै कम्त पाँच
  - Word errors: i→∅; want→ए; to→पाउँदा; check→तत्साक; my→नै; account→कम्त; balance→पाँच

- **p-conv-01** (conversational)
  - Prediction: अब तिहार मलाई लाग्छ किबी अलि बढी आयो जस्तो छ
  - Word errors: अँ→अब; त्यो→तिहार; कि→∅; बिल→किबी

### missed:names

- **p-names-01** (names)
  - Prediction: मेरो नाम राम बहादुर तापा हो
  - Word errors: थापा→तापा

- **p-names-02** (names)
  - Prediction: सीता कुमारी श्रेष्ठलाई फोन गरिदिनुहोस्
  - Word errors: 

- **p-proper-01** (proper_nouns)
  - Prediction: म फोकाबाट लुम्बिने जाँदैछु
  - Word errors: पोखराबाट→फोकाबाट; लुम्बिनी→लुम्बिने

### missed:currency

- **p-currency-01** (currency)
  - Prediction: मेरो बास पच्चिस हजार छ
  - Word errors: balance→बास; रु→पच्चिस; 25000→हजार

- **p-currency-02** (currency)
  - Prediction: मैले पन्ध्र सय रुपियाँ तिरेँ
  - Word errors: रुपैयाँ→रुपियाँ

### missed:addresses

- **p-address-01** (addresses)
  - Prediction: म बानेस्वर कार्टमा रोमओर नम्बर दसमा बस्छु
  - Word errors: बानेश्वर→बानेस्वर; काठमाडौं→कार्टमा; वडा→रोमओर

- **p-address-02** (addresses)
  - Prediction: हाम्रो घर फुटसोकै ललितपुरमा छ
  - Word errors: पुल्चोक→फुटसोकै

### missed:phone_numbers

- **p-numbers-01** (phone_numbers)
  - Prediction: मेरो फोन नम्बर नभो अरब चौरासी करोड बाह्र लाख चौतिस हजार पाँच सय सतसट्ठी हो
  - Word errors: ∅→नभो; ∅→अरब; ∅→चौरासी; ∅→करोड; ∅→बाह्र; ∅→लाख; ∅→चौतिस; ∅→हजार

### missed:otps

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

### missed:account_numbers

- **p-numbers-03** (account_numbers)
  - Prediction: मेरो खाता नम्बर शून्य एक दुई तिन चार पाँच छ सात आठ नौ हो
  - Word errors: ∅→शून्य; ∅→एक; ∅→दुई; ∅→तिन; ∅→चार; ∅→पाँच; ∅→छ; ∅→सात

### missed:numbers

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

### missed:dates

- **p-dates-01** (dates)
  - Prediction: मेरो पृथ्वी मन्त्र दुरी हजार त्रासी साल असोस बार गति हो
  - Word errors: ∅→पृथ्वी; ∅→मन्त्र; ∅→दुरी; appointment→हजार; 2083→त्रासी; असोज→असोस; 12→बार; गते→गति

## indicconformer-600m [original]

Error tag counts (samples containing each error type): substitution=21, devanagari=15, code_switching=9, english_word=9, missed:mixed_language=9, insertion=7, number=4, deletion=3, missed:names=3, missed:currency=2, missed:phone_numbers=1, missed:otps=1, missed:account_numbers=1, missed:numbers=1, missed:dates=1, missed:addresses=1

### substitution

- **p-basic-01** (basic_nepali)
  - Prediction: नमस्ते मलाई तपाइनको सहयोग चाहियो
  - Word errors: तपाईंको→तपाइनको

- **p-devanagari-01** (devanagari)
  - Prediction: कृपया मेरो गुना सुधरता गरिदिनुहोस्
  - Word errors: गुनासो→गुना; दर्ता→सुधरता

- **p-conjunct-02** (conjunct_characters)
  - Prediction: श्रमाको स्वास्थ्य परीक्षण द्वितीय पटक गरियो
  - Word errors: श्रीमान्को→श्रमाको

### devanagari

- **p-basic-01** (basic_nepali)
  - Prediction: नमस्ते मलाई तपाइनको सहयोग चाहियो
  - Word errors: तपाईंको→तपाइनको

- **p-devanagari-01** (devanagari)
  - Prediction: कृपया मेरो गुना सुधरता गरिदिनुहोस्
  - Word errors: गुनासो→गुना; दर्ता→सुधरता

- **p-conjunct-02** (conjunct_characters)
  - Prediction: श्रमाको स्वास्थ्य परीक्षण द्वितीय पटक गरियो
  - Word errors: श्रीमान्को→श्रमाको

### code_switching

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-currency-01** (currency)
  - Prediction: मेरो बास्र पच्चिस हार छ
  - Word errors: balance→बास्र; रु→पच्चिस; 25000→हार

### english_word

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-currency-01** (currency)
  - Prediction: मेरो बास्र पच्चिस हार छ
  - Word errors: balance→बास्र; रु→पच्चिस; 25000→हार

### missed:mixed_language

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-currency-01** (currency)
  - Prediction: मेरो बास्र पच्चिस हार छ
  - Word errors: balance→बास्र; रु→पच्चिस; 25000→हार

### insertion

- **p-numbers-01** (phone_numbers)
  - Prediction: मेरो फोन नम्बर नव अर चौरासी करोड बाह्र लाख चौतिस हजार पाँच सय सत्सट्ठी हो
  - Word errors: ∅→नव; ∅→अर; ∅→चौरासी; ∅→करोड; ∅→बाह्र; ∅→लाख; ∅→चौतिस; ∅→हजार

- **p-numbers-03** (account_numbers)
  - Prediction: मेरो खाता नम्बर शून्य एक दुई तिन चार पाँच छ सात आठ नम्ब हो
  - Word errors: ∅→शून्य; ∅→एक; ∅→दुई; ∅→तिन; ∅→चार; ∅→पाँच; ∅→छ; ∅→सात

- **p-dates-01** (dates)
  - Prediction: मेरो प्रृतमञ्च धुरी हजार त्रियासी साल असोस बारगति हो
  - Word errors: ∅→प्रृतमञ्च; appointment→धुरी; 2083→हजार; साल→त्रियासी; असोज→साल; 12→असोस; गते→बारगति

### number

- **p-numbers-01** (phone_numbers)
  - Prediction: मेरो फोन नम्बर नव अर चौरासी करोड बाह्र लाख चौतिस हजार पाँच सय सत्सट्ठी हो
  - Word errors: ∅→नव; ∅→अर; ∅→चौरासी; ∅→करोड; ∅→बाह्र; ∅→लाख; ∅→चौतिस; ∅→हजार

- **p-numbers-03** (account_numbers)
  - Prediction: मेरो खाता नम्बर शून्य एक दुई तिन चार पाँच छ सात आठ नम्ब हो
  - Word errors: ∅→शून्य; ∅→एक; ∅→दुई; ∅→तिन; ∅→चार; ∅→पाँच; ∅→छ; ∅→सात

- **p-currency-01** (currency)
  - Prediction: मेरो बास्र पच्चिस हार छ
  - Word errors: balance→बास्र; रु→पच्चिस; 25000→हार

### deletion

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

- **p-names-01** (names)
  - Prediction: मेरो नाम रामबहादुर थापा हो
  - Word errors: राम→∅; बहादुर→रामबहादुर

- **p-conv-01** (conversational)
  - Prediction: अब तिहार मलाई लाग्छ किबी अलि बढी आयो जस्तो छ
  - Word errors: अँ→अब; त्यो→तिहार; कि→∅; बिल→किबी

### missed:names

- **p-names-01** (names)
  - Prediction: मेरो नाम रामबहादुर थापा हो
  - Word errors: राम→∅; बहादुर→रामबहादुर

- **p-names-02** (names)
  - Prediction: सीता कुमारी श्रेष्ठलाई फोन गरिदिनुहोस्
  - Word errors: 

- **p-proper-01** (proper_nouns)
  - Prediction: म पोखबाट लुम्बि नै जाँदैछु
  - Word errors: ∅→पोखबाट; पोखराबाट→लुम्बि; लुम्बिनी→नै

### missed:currency

- **p-currency-01** (currency)
  - Prediction: मेरो बास्र पच्चिस हार छ
  - Word errors: balance→बास्र; रु→पच्चिस; 25000→हार

- **p-currency-02** (currency)
  - Prediction: मैले पन्ध्र सय रुपियाँ तरेँ
  - Word errors: रुपैयाँ→रुपियाँ; तिरेँ→तरेँ

### missed:phone_numbers

- **p-numbers-01** (phone_numbers)
  - Prediction: मेरो फोन नम्बर नव अर चौरासी करोड बाह्र लाख चौतिस हजार पाँच सय सत्सट्ठी हो
  - Word errors: ∅→नव; ∅→अर; ∅→चौरासी; ∅→करोड; ∅→बाह्र; ∅→लाख; ∅→चौतिस; ∅→हजार

### missed:otps

- **p-numbers-02** (otps)
  - Prediction: मेरो अतिथि चार सात दुई नौ एक पाँच हो
  - Word errors: otp→अतिथि

### missed:account_numbers

- **p-numbers-03** (account_numbers)
  - Prediction: मेरो खाता नम्बर शून्य एक दुई तिन चार पाँच छ सात आठ नम्ब हो
  - Word errors: ∅→शून्य; ∅→एक; ∅→दुई; ∅→तिन; ∅→चार; ∅→पाँच; ∅→छ; ∅→सात

### missed:numbers

- **p-numbers-04** (numbers)
  - Prediction: मलाई तिनवटा कनेक्सन चाहियो
  - Word errors: तीन→∅; वटा→तिनवटा; connection→कनेक्सन

### missed:dates

- **p-dates-01** (dates)
  - Prediction: मेरो प्रृतमञ्च धुरी हजार त्रियासी साल असोस बारगति हो
  - Word errors: ∅→प्रृतमञ्च; appointment→धुरी; 2083→हजार; साल→त्रियासी; असोज→साल; 12→असोस; गते→बारगति

### missed:addresses

- **p-address-02** (addresses)
  - Prediction: हाम्रो घर फुर्सोकै ललितपुरमा छ
  - Word errors: पुल्चोक→फुर्सोकै

