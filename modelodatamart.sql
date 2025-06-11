-- ================================
-- DATAMART DE MERMAS 
-- ================================

-- DIMENSIÓN TIENDA
CREATE TABLE TIENDA (
    id_tienda INT PRIMARY KEY,
    Nombre VARCHAR(100),
    Comuna VARCHAR(50),
    Region VARCHAR(50),
    Ciudad VARCHAR(50),
    Zonal VARCHAR(50)
);

-- DIMENSIÓN TIEMPO
CREATE TABLE TIEMPO (
    id_tiempo INT PRIMARY KEY,
    Dia INT,
    Mes INT,
    Ano INT,
    Semestre INT,
    Trimestre INT,
    Semana INT,
    Dia_Semana INT,
    Temporada VARCHAR(20),
    Fin_de_semana VARCHAR(10),
    Fecha DATE
);

-- DIMENSIÓN PRODUCTO
CREATE TABLE PRODUCTO (
    id_producto INT PRIMARY KEY,
    Codigo_producto VARCHAR(50),
    Descripcion VARCHAR(200),
    Negocio VARCHAR(100),
    Seccion VARCHAR(100),
    Linea VARCHAR(100),
    Categoria VARCHAR(100),
    Abastecimiento VARCHAR(100),
    Tipo_empaque VARCHAR(50),
    Vida_util INT,
    Perecibilidad VARCHAR(20),
    Demanda VARCHAR(20),
    Tipo_producto VARCHAR(50)
);

-- DIMENSIÓN MOTIVO
CREATE TABLE MOTIVO (
    id_motivo INT PRIMARY KEY,
    Tipo_motivo VARCHAR(100),
    Ubicacion_motivo VARCHAR(100),
    Motivo VARCHAR(200)
);

-- TABLA DE HECHOS MERMAS
CREATE TABLE MERMAS (
    id_merma INT PRIMARY KEY,
    id_tiempo INT,
    id_producto INT,
    id_tienda INT,
    id_motivo INT,
    Merma_unidad DECIMAL(10,2),
    Merma_monto DECIMAL(12,2),
    
    FOREIGN KEY (id_tiempo) REFERENCES TIEMPO(id_tiempo),
    FOREIGN KEY (id_producto) REFERENCES PRODUCTO(id_producto),
    FOREIGN KEY (id_tienda) REFERENCES TIENDA(id_tienda),
    FOREIGN KEY (id_motivo) REFERENCES MOTIVO(id_motivo)
);

-- ÍNDICES BÁSICOS
CREATE INDEX idx_mermas_tiempo ON MERMAS(id_tiempo);
CREATE INDEX idx_mermas_tienda ON MERMAS(id_tienda);
CREATE INDEX idx_mermas_producto ON MERMAS(id_producto);
CREATE INDEX idx_mermas_motivo ON MERMAS(id_motivo);