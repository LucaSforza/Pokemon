create domain String as varchar(100);

create domain IntegerGEZ as integer
    check(value >= 0);

create type DatasetType as enum('Train','Test');