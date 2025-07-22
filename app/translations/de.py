translations = {
    # Hauptelemente
    'app_title': 'Intelligentes Hautkrebsdiagnosesystem',
    'app_description': 'Dieses System verwendet **speziell für Hautkrebs trainierte Modelle** mit dem ISIC 2019-Datensatz.',
    'settings': '⚙️ Einstellungen',
    'settings_description': 'Analyseparameter',
    'image_upload': '📸 Bild hochladen',
    
    # PDF spezifisch
    'pdf_report_title': 'Intelligenter Medizinischer Bericht',
    'report_date_time': 'Analysedatum und -zeit',
    'model_used': 'Verwendetes Modell',
    'threshold_value': 'Vertrauensschwelle',
    'analyzed_image': 'ANALYSIERTES BILD',
    'diagnosis_results': 'DIAGNOSEERGEBNISSE',
    'raw_confidence_value': 'Rohwert des Modells',
    'raw_value': 'Rohwert',
    
    # Konfiguration
    'debug_mode': '🐛 Debug-Modus',
    'debug_help': 'Aktiviert detaillierte Infos',
    'select_model': '🤖 Modell auswählen',
    'select_model_help': 'Verschiedene Leistungsmerkmale',
    'model_info': '📊 **Modellinformationen:**',
    'parameters': '**Parameter:**',
    'layers': '**Schichten:**',
    'hybrid_models': '🧠 Erweiterte Hybridmodelle',
    'hybrid_models_available': '✅ Verfügbare Hybridmodelle:',
    'no_hybrid_models': '⚠️ Keine Hybridmodelle erkannt',
    'train_hybrid_models': '🚀 Hybridmodelle trainieren',
    'train_hybrid_help': 'Startet das Training erweiterter Hybridmodelle. Dieser Prozess kann mehrere Stunden dauern.',
    'training_in_progress': 'Training läuft...',
    'training_completed': '✅ Training abgeschlossen!',
    'training_error': '❌ Fehler:',
    'reloading': '🔄 Neu laden, um neue Modelle zu erkennen...',
    'confidence_threshold': '🎯 Vertrauensschwelle',
    'confidence_help': 'Höhere Werte = mehr Vertrauen',
    'decision_threshold': '⚖️ Bösartig/Gutartig-Schwelle',
    'decision_help': 'Niedrigere Werte = empfindlicher',
    'threshold_note': '💡 **Hinweis**: Niedrigere Schwelle erhöht die Empfindlichkeit.',
    
    # Bilder
    'upload_prompt': 'Hautläsionsbild hochladen (JPG, JPEG, PNG)',
    'upload_help': 'Bild sollte klar sein',
    'original_image': 'Originalbild',
    'processed_image': 'Verarbeitetes Bild (300x300)',
    
    # Ergebnisse
    'processing_image': 'Verarbeitung...',
    'benign': 'Gutartig',
    'malignant': 'Bösartig',
    'confidence': 'Vertrauen',
    'prediction': 'Diagnose',
    'diagnosis_results': 'Diagnoseergebnisse',
    'advanced_analysis': 'Erweiterte Analyse',
    'metrics_title': 'Metriken',
    'attention_required': '🚨 **Aufmerksamkeit erforderlich**: Das System hat Merkmale erkannt, die auf eine bösartige Läsion hindeuten. Es wird empfohlen, **dringend** einen Spezialisten zu konsultieren.',
    'model_comparison_desc': 'Ergebnisse der Analyse des gleichen Bildes mit verschiedenen Modellen',
    
    # Basismetriken
    'accuracy': 'Genauigkeit',
    'sensitivity': 'Sensitivität',
    'specificity': 'Spezifität',
    
    # Sprachen
    'language': 'Sprache',
    
    # Zusätzliche Abschnitte
    'results_interpretation': 'Ergebnisinterpretation',
    'model_comparison': 'Modellvergleich',
    'consistency_analysis': 'Konsistenzanalyse',
    'pdf_success': 'PDF-Bericht erfolgreich erstellt',
    
    # Warnungen und Nachrichten
    'low_confidence_warning': '⚠️ **Geringes Vertrauen**: Konsultieren Sie einen Spezialisten.',
    'favorable_result': '✅ **Günstiges Ergebnis**: Die Läsion scheint gutartig zu sein.',
    'attention_required': '🚨 **Aufmerksamkeit erforderlich**: Bösartige Merkmale erkannt.',
    
    # Technische Informationen und rechtliche Hinweise
    'technical_info': 'TECHNISCHE INFORMATIONEN',
    'technical_dataset': 'Datensatz: ISIC 2019 (25.331 reale Bilder)',
    'technical_type': 'Typ: Binäre Klassifikation (Gutartig/Bösartig)',
    'technical_accuracy': 'Genauigkeit: ~69% (optimiert für Hautkrebs)',
    'technical_input': 'Eingabe: 300x300 Pixel',
    'technical_architecture': 'Architektur: Transfer Learning mit Fine-Tuning',
    'medical_disclaimer_title': 'MEDIZINISCHER HAFTUNGSAUSSCHLUSS',
    'medical_disclaimer_1': 'Dieses System dient nur zu Bildungs- und Forschungszwecken.',
    'medical_disclaimer_2': 'Die Ergebnisse stellen KEINE medizinische Diagnose dar.',
    'medical_disclaimer_3': 'Konsultieren Sie IMMER einen Dermatologen für eine professionelle Diagnose.',
    
    # PDF und Berichte
    'pdf_section_title': 'PDF-Bericht erstellen',
    'generate_pdf_button': 'Vollständigen PDF-Bericht generieren',
    'pdf_includes': 'Der PDF-Bericht enthält',
    'pdf_content_diagnosis': 'Diagnose und Bildanalyse',
    'pdf_content_comparison': 'Vergleich aller Modelle',
    'pdf_content_matrix': 'Konfusionsmatrix und erweiterte Metriken',
    'pdf_content_charts': 'MCC-Diagramme und statistische Analyse',
    'pdf_content_mcnemar': 'McNemar-Tests',
    'pdf_content_recommendations': 'Medizinische Empfehlungen',
    
    # Bezeichnungen für Diagramme und statistische Analyse
    'confusion_matrix_title': 'Konfusionsmatrix',
    'metrics_dashboard_title': 'Metrik-Dashboard',
    'statistical_analysis_title': 'Erweiterte statistische Analyse',
    'model_analysis_description': 'Detaillierte Analyse der Leistung des ausgewählten Modells',
    'statistical_analysis_description': 'Einschließlich Matthews-Koeffizient und McNemar-Tests',
    'mcc_comparison_title': 'Vergleichende Übersicht der Matthews-Koeffizienten (MCC)',
    'mcc_comparison_description': 'Vergleich aller Modelle basierend auf dem Matthews-Koeffizienten',
    'mcnemar_tests_title': 'McNemar Statistische Tests',
    'mcnemar_description': 'Statistischer Vergleich zwischen Modellen',
    'activation_maps_title': 'Visualisierung der Aktivierungskarten',
    'activation_maps_description': 'Visualisierung der Regionen, die die Diagnose am stärksten beeinflusst haben',
    'generating_activation_map': 'Aktivierungskarte wird generiert...',
    'activation_map_caption': 'Aktivierungskarte (Grad-CAM)',
    'heatmap_description': 'Die Heatmap zeigt die Regionen, die die Diagnose des Modells am stärksten beeinflusst haben. Rote und gelbe Bereiche sind am relevantesten.',
    'activation_error': 'Die Aktivierungskarte konnte für dieses Modell nicht generiert werden.',
    
    # Metrikinterpretation
    'metrics_interpretation': '📋 Interpretation:',
    'accuracy_explanation': 'der Vorhersagen sind korrekt',
    'sensitivity_explanation': 'der malignen Fälle werden erkannt',
    'specificity_explanation': 'der benignen Fälle werden korrekt identifiziert',
    'precision_explanation': 'der als maligne eingestuften Fälle sind tatsächlich maligne',
    'f1_explanation': 'ist das Gleichgewicht zwischen Präzision und Sensitivität',
    'mcc_explanation': '(Matthews-Korrelationskoeffizient - ausgewogen für ungleiche Klassen)',
    
    # Metriktabelle und andere Etiketten
    'confusion_matrix_chart': '🎯 Konfusionsmatrix',
    'advanced_metrics': '📈 Erweiterte Leistungsmetriken',
    'mcc_table_title': '📋 Übersichtstabelle - Matthews-Koeffizienten',
    'generating_pdf': 'PDF-Bericht wird generiert...',
    'real_data_metrics': '✅ **Reale Trainingsdaten**: Anzeige realer Metriken für das Modell {model} im ISIC 2019-Datensatz',
    'simulated_data_metrics': '⚠️ **Simulierte Daten**: Verwendung von Beispieldaten zur Demonstration',
    'mcnemar_test_results': 'McNemar-Testergebnisse',
    
    # PDF-Diagrammbezeichnungen
    'confidence_comparison_plot': 'Vertrauensvergleich',
    'inference_speed_plot': 'Inferenzgeschwindigkeit',
    'mcc_comparative_plot': 'MCC-Vergleich',
    'mcnemar_pvalues_plot': 'McNemar P-Werte',
    
    # Konsistenzanalyse
    'perfect_consistency': '✅ **Perfekte Konsistenz**: Alle Modelle stimmen in der Diagnose überein:',
    'inconsistency_detected': '⚠️ **Inkonsistenz festgestellt**: Die Modelle stimmen nicht in der Diagnose überein',
    'diagnoses_obtained': '**Erhaltene Diagnosen**:',
    'recommendation_title': '💡 **Empfehlung**:',
    'inconsistency_recommendation': 'Bei Unstimmigkeiten wird empfohlen, einen Spezialisten zur Bestätigung zu konsultieren.',
    
    # Konfusionsmatrix-Interpretation
    'confusion_matrix_interpretation': '🔍 Interpretation der Konfusionsmatrix',
    'matrix_elements': '**📊 Matrixelemente:**',
    'true_positives': '**Richtig Positive (TP)**: Korrekt identifizierte maligne Fälle',
    'true_negatives': '**Richtig Negative (TN)**: Korrekt identifizierte benigne Fälle',
    'false_positives': '**Falsch Positive (FP)**: Benigne Fälle, die als maligne klassifiziert wurden',
    'false_negatives': '**Falsch Negative (FN)**: Maligne Fälle, die als benigne klassifiziert wurden',
    'medical_importance': '**🎯 Medizinische Bedeutung:**',
    'fn_critical': '**Falsch Negative** sind kritisch (unentdeckter Krebs)',
    'fp_anxiety': '**Falsch Positive** verursachen unnötige Angst',
    'recall_importance': '**Hoher Recall** ist entscheidend für die Früherkennung',
    'precision_importance': '**Hohe Präzision** reduziert Fehlalarme',
    
    # MCC-Interpretation
    'efficientnet_title': '**🥇 EfficientNetB4:**',
    'efficientnet_metrics': '- MCC: 0.7845 (**Ausgezeichnet**)\n- Beste Gesamtbalance\n- Empfohlen für klinischen Einsatz\n- Überlegene diagnostische Zuverlässigkeit',
    'resnet_title': '**🥈 ResNet152:**',
    'resnet_metrics': '- MCC: 0.6234 (**Gut**)\n- Moderate Leistung\n- Praktikable Alternative\n- Akzeptables Gleichgewicht',
    'custom_cnn_title': '**🥉 Benutzerdefiniertes CNN:**',
    'custom_cnn_metrics': '- MCC: 0.5789 (**Gut**)\n- Standardleistung\n- Komplementäre Option\n- Mögliche Verbesserungen',
    
    # Modellbewertung und Statistik
    'best_balance': 'Beste Gesamtbalance',
    'recommended_clinical': 'Empfohlen für klinischen Einsatz',
    'superior_reliability': 'Überlegene diagnostische Zuverlässigkeit',
    'moderate_performance': 'Moderate Leistung',
    'viable_alternative': 'Praktikable Alternative',
    'acceptable_balance': 'Akzeptables Gleichgewicht',
    'standard_performance': 'Standardleistung',
    'complementary_option': 'Komplementäre Option',
    'possible_improvements': 'Mögliche Verbesserungen',
    'statistical_conclusions': 'Statistische Schlussfolgerungen',
    'statistical_superiority': 'zeigt signifikante statistische Überlegenheit',
    'superior_comparisons': '**Vergleiche, bei denen {model} überlegen ist:**',
    'no_statistical_diff': 'Keine statistisch signifikanten Unterschiede zwischen den Modellen',
    'medical_interpretation': 'Medizinische Interpretation',
    'for_model': 'für',
    'mcnemar_confirm': 'Die McNemar-Testergebnisse bestätigen, dass',
    'stat_diff': 'Zeigt statistisch signifikante Unterschiede im Vergleich zu anderen Modellen',
    'diagnostic_superiority': 'Zeigt Überlegenheit bei der diagnostischen Genauigkeit',
    'clinical_reliability': 'Bietet größere Zuverlässigkeit für klinische Entscheidungen',
    'robust_option': 'Ist die robusteste Option für die medizinische Implementierung',
    'justified_selection': 'Rechtfertigt seine Auswahl als Hauptmodell für die Diagnose',
}
