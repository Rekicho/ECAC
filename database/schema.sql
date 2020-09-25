PRAGMA foreign_key = on;

DROP TABLE IF EXISTS District;

CREATE TABLE District (
    code                    INTEGER PRIMARY KEY,
    name                    TEXT,
    region                  TEXT,
    inhabitants             INTEGER,
    municipalities_499      INTEGER,
    municipalities_1999     INTEGER,
    municipalities_9999     INTEGER,
    municipalities_max      INTEGER,
    cities                  INTEGER,
    ratio_urban_inhabitants FLOAT,
    average_salary          INTEGER,
    unemployment_rate_95    FLOAT,
    unemployment_rate_96    FLOAT,
    number_enterpreneurs    INTEGER,
    comminted_crimes_95     INTEGER,
    comminted_crimes_96     INTEGER
);

DROP TABLE IF EXISTS Account;

CREATE TABLE Account (
    account_id          INTEGER PRIMARY KEY,
    district_id         INTEGER REFERENCES District,
    frequency           TEXT,
    date                TEXT
);

DROP TABLE IF EXISTS Client;

CREATE TABLE Client (
    client_id           INTEGER PRIMARY KEY,
    birth_numver        INTEGER,
    district_id         INTEGER REFERENCES District
);

DROP TABLE IF EXISTS Disposition;

CREATE TABLE Disposition (
    disp_id             INTEGER PRIMARY KEY,
    client_id           INTEGER REFERENCES Client,
    account_id          INTEGER REFERENCES Account,
    type                TEXT
);

DROP TABLE IF EXISTS Trans;

CREATE TABLE Trans (
    trans_id            INTEGER PRIMARY KEY,
    account_id          INTEGER REFERENCES Account,
    date                TEXT,
    type                TEXT,
    operation           TEXT,
    amount              FLOAT,
    balance             FLOAT,
    k_symbol            TEXT,
    bank                TEXT,
    account             INTEGER    
);

DROP TABLE IF EXISTS Loan;

CREATE TABLE Loan (
    loan_id             INTEGER PRIMARY KEY,
    account_id          INTEGER REFERENCES Account,
    date                TEXT,
    amount              FLOAT,
    payments            FLOAT,
    status              INTEGER
);

DROP TABLE IF EXISTS Card;

CREATE TABLE Card (
    card_id             INTEGER PRIMARY KEY,
    disp_id             INTEGER REFERENCES Disposition,
    type                INTEGER,
    issued              TEXT
);

