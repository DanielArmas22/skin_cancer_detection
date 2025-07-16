import streamlit as st

st.title("Prueba de Streamlit")
st.write("Si ves esto, Streamlit está funcionando correctamente.")

# Probar importación del sistema
try:
    from system_manager import SkinCancerDiagnosisSystem
    st.success("✅ Sistema manager importado correctamente")
    
    # Probar inicialización
    system = SkinCancerDiagnosisSystem()
    st.success("✅ Sistema inicializado")
    
    # Probar estado
    if system.initialize_components():
        st.success("✅ Componentes inicializados")
        status = system.get_system_status()
        st.json(status)
    else:
        st.error("❌ Error inicializando componentes")
        
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())
