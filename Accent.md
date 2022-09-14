package com.example.mainproject;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    public Button give_speech;
    public Button give_text;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        give_speech = findViewById(R.id.give_speech);
        give_text = findViewById(R.id.give_text);
    }

        public void SpeechClick(View view){
        Intent intent1 = new Intent(this,givespeech.class);
        startActivity(intent1);
        }

        public void TypeClick(View view){
        Intent intent2 = new Intent(this,givetext.class);
        startActivity(intent2);
        }
    }
